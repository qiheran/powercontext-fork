# Copyright (c) 2026 OceanBase.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Concurrent embedding request coalescing with a bounded vector cache.

``CoalescingEmbeddingModel`` wraps any :class:`EmbeddingModel` and coordinates
cross-call work sharing:

- Texts repeated inside one call, across concurrent calls inside the coalesce
  window, and across calls that arrive while a request is still executing are
  sent to the wrapped model exactly once.
- Successful vectors are memoized in a bounded LRU cache (``cache_size=0``
  disables cross-call caching while in-flight sharing stays active).
- Every caller keeps its original input order and duplicate positions, and
  each caller's ``usage`` aggregates the provider usage of exactly the
  distinct underlying batches it waited on.
- Callers are cancellation-isolated: cancelling one waiter never cancels the
  shared underlying request nor steals results from other waiters. Failures
  are delivered only to the waiters of the failing batch and are never cached.
"""

from __future__ import annotations

import asyncio
from collections import OrderedDict
from collections.abc import Awaitable, Callable
from typing import TypeAlias

from powercontext.builtin.inference.models import EmbeddingResult, EmbeddingVector, InferenceUsage
from powercontext.builtin.inference.protocols import EmbeddingModel

__all__ = ["CoalescingEmbeddingModel"]

_VectorSource: TypeAlias = Callable[[tuple[str, ...]], Awaitable[EmbeddingResult]]


def _merge_tokens(left: int | None, right: int | None) -> int | None:
    """Sum optional token counters, preserving ``None`` when nothing was seen."""

    if left is None:
        return right
    if right is None:
        return left
    return left + right


class _Batch:
    """One dispatched request against the wrapped model."""

    __slots__ = ("texts",)

    def __init__(self, texts: tuple[str, ...]) -> None:
        self.texts = texts


class _Entry:
    """Shared per-text coordination state (pending or attached to a batch)."""

    __slots__ = ("batch", "waiters")

    def __init__(self) -> None:
        self.batch: _Batch | None = None
        self.waiters: list[tuple[_Request, str]] = []


class _Request:
    """One caller's in-flight ``embed`` call."""

    __slots__ = (
        "charged",
        "future",
        "input_tokens",
        "output_tokens",
        "remaining",
        "requests",
        "slots",
        "slot_for_text",
    )

    def __init__(
        self,
        slots: list[EmbeddingVector | None],
        slot_for_text: dict[str, list[int]],
        future: asyncio.Future[EmbeddingResult],
    ) -> None:
        self.slots = slots
        self.slot_for_text = slot_for_text
        self.future = future
        self.remaining = len(slot_for_text)
        self.requests = 0
        self.input_tokens: int | None = None
        self.output_tokens: int | None = None
        self.charged: set[_Batch] = set()


class CoalescingEmbeddingModel:
    """Merge concurrent embedding requests and cache successful vectors."""

    def __init__(
        self,
        model: EmbeddingModel,
        *,
        max_batch_size: int = 64,
        coalesce_ms: float = 5.0,
        cache_size: int = 1024,
    ) -> None:
        if max_batch_size < 1:
            raise ValueError("max_batch_size must be at least 1")
        if coalesce_ms < 0:
            raise ValueError("coalesce_ms must not be negative")
        if cache_size < 0:
            raise ValueError("cache_size must not be negative")
        self._model = model
        self.profile = model.profile
        self._embed: _VectorSource = model.embed
        self._max_batch_size = max_batch_size
        self._coalesce_seconds = coalesce_ms / 1000.0
        self._cache_size = cache_size
        self._cache: OrderedDict[str, EmbeddingVector] = OrderedDict()
        self._entries: dict[str, _Entry] = {}
        self._wake = asyncio.Event()
        self._dispatcher: asyncio.Task[None] | None = None
        self._batches: set[asyncio.Task[None]] = set()

    async def embed(self, texts: tuple[str, ...], /) -> EmbeddingResult:
        """Embed ``texts``, sharing concurrent work and reusing cached vectors."""

        order = tuple(texts)
        if not order:
            return EmbeddingResult(vectors=(), usage=InferenceUsage(requests=0))

        slots: list[EmbeddingVector | None] = [None] * len(order)
        slot_for_text: dict[str, list[int]] = {}
        pending: list[str] = []
        for index, text in enumerate(order):
            cached = self._cache_get(text)
            if cached is not None:
                slots[index] = cached
                continue
            positions = slot_for_text.get(text)
            if positions is None:
                positions = []
                slot_for_text[text] = positions
                pending.append(text)
            positions.append(index)

        if not pending:
            return EmbeddingResult(vectors=tuple(slots), usage=InferenceUsage(requests=0))

        future: asyncio.Future[EmbeddingResult] = asyncio.get_running_loop().create_future()
        request = _Request(slots=slots, slot_for_text=slot_for_text, future=future)
        wake = False
        for text in pending:
            entry = self._entries.get(text)
            if entry is None:
                entry = _Entry()
                self._entries[text] = entry
                wake = True
            elif entry.batch is None:
                wake = True
            entry.waiters.append((request, text))
        if wake:
            self._ensure_dispatcher()
        return await future

    async def embed_query(self, texts: tuple[str, ...], /) -> EmbeddingResult:
        """Embed retrieval queries through the same coalescing pipeline."""

        return await self.embed(texts)

    # ------------------------------------------------------------------
    # LRU cache
    # ------------------------------------------------------------------

    def _cache_get(self, text: str) -> EmbeddingVector | None:
        if self._cache_size <= 0:
            return None
        vector = self._cache.get(text)
        if vector is not None:
            self._cache.move_to_end(text)
        return vector

    def _cache_put(self, text: str, vector: EmbeddingVector) -> None:
        if self._cache_size <= 0:
            return
        self._cache[text] = vector
        self._cache.move_to_end(text)
        while len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)

    # ------------------------------------------------------------------
    # Dispatching
    # ------------------------------------------------------------------

    def _ensure_dispatcher(self) -> None:
        if self._dispatcher is None or self._dispatcher.done():
            self._dispatcher = asyncio.create_task(self._run_dispatcher())
        self._wake.set()

    async def _run_dispatcher(self) -> None:
        while True:
            await self._wake.wait()
            self._wake.clear()
            if self._coalesce_seconds > 0:
                await asyncio.sleep(self._coalesce_seconds)
            else:
                # Yield one loop tick so callers that are already runnable in
                # this tick can join the same batch even with a zero window.
                await asyncio.sleep(0)
            self._dispatch_pending()

    def _dispatch_pending(self) -> None:
        ready = tuple((text, entry) for text, entry in self._entries.items() if entry.batch is None)
        if not ready:
            return
        for start in range(0, len(ready), self._max_batch_size):
            chunk = ready[start : start + self._max_batch_size]
            batch = _Batch(texts=tuple(text for text, _ in chunk))
            for _, entry in chunk:
                entry.batch = batch
            task = asyncio.create_task(self._run_batch(batch))
            self._batches.add(task)
            task.add_done_callback(self._batches.discard)

    async def _run_batch(self, batch: _Batch) -> None:
        try:
            result = await self._embed(batch.texts)
            vectors = tuple(result.vectors)
            if len(vectors) != len(batch.texts):
                raise ValueError(
                    "embedding model returned "
                    f"{len(vectors)} vectors for {len(batch.texts)} batch texts"
                )
        except Exception as error:
            self._fail_batch(batch, error)
            return
        self._complete_batch(batch, vectors, result.usage)

    def _complete_batch(
        self,
        batch: _Batch,
        vectors: tuple[EmbeddingVector, ...],
        usage: InferenceUsage,
    ) -> None:
        for text, vector in zip(batch.texts, vectors, strict=True):
            self._cache_put(text, vector)
        for text, vector in zip(batch.texts, vectors, strict=True):
            entry = self._entries.pop(text, None)
            if entry is None or entry.batch is not batch:
                continue
            for request, _ in entry.waiters:
                if request.future.done():
                    continue
                for position in request.slot_for_text[text]:
                    request.slots[position] = vector
                if batch not in request.charged:
                    request.charged.add(batch)
                    request.requests += usage.requests
                    request.input_tokens = _merge_tokens(request.input_tokens, usage.input_tokens)
                    request.output_tokens = _merge_tokens(request.output_tokens, usage.output_tokens)
                request.remaining -= 1
                if request.remaining <= 0:
                    request.future.set_result(
                        EmbeddingResult(
                            vectors=tuple(request.slots),
                            usage=InferenceUsage(
                                requests=request.requests,
                                input_tokens=request.input_tokens,
                                output_tokens=request.output_tokens,
                            ),
                        )
                    )

    def _fail_batch(self, batch: _Batch, error: Exception) -> None:
        for text in batch.texts:
            entry = self._entries.pop(text, None)
            if entry is None or entry.batch is not batch:
                continue
            for request, _ in entry.waiters:
                if not request.future.done():
                    request.future.set_exception(error)
