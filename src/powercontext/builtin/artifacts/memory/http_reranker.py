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

"""HTTP scoring-service reranking for coarse Memory search candidates.

The adapter keeps the :class:`MemoryReranker` contract: it splits the coarse
pool into bounded batches, POSTs each batch to the configured endpoint, maps
the returned scores back onto the original one-based global ranks, and selects
the final ranks with one stable global ordering (score descending, original
rank ascending on ties). Candidate identity is never rewritten — the caller
resolves selected ranks against the original ``MemoryHit`` pool.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from typing import Any

import httpx

from powercontext.builtin.artifacts.memory.models import MemoryHit
from powercontext.builtin.artifacts.memory.reranking import MemoryRerankDecision
from powercontext.builtin.inference import InferenceUsage

DEFAULT_HTTP_RERANK_POLICY_ID = "http-cross-encoder-v1"
DEFAULT_HTTP_RERANK_BATCH_SIZE = 32

_SCORE_RECORD_KEYS = frozenset({"rank", "score"})


class HTTPMemoryReranker:
    """Rank Memory hits with an external HTTP scoring service.

    Requests use the caller-provided ``httpx.AsyncClient``; every candidate is
    sent exactly once under its original one-based global rank, in batches of
    at most ``batch_size`` candidates. Transport failures and non-2xx
    responses surface as their original ``httpx`` exceptions; malformed
    payloads raise ``ValueError``.
    """

    policy_id: str

    def __init__(
        self,
        endpoint: str,
        *,
        client: httpx.AsyncClient,
        batch_size: int = DEFAULT_HTTP_RERANK_BATCH_SIZE,
        policy_id: str = DEFAULT_HTTP_RERANK_POLICY_ID,
    ) -> None:
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")  # noqa: TRY003
        self._endpoint = endpoint
        self._client = client
        self._batch_size = batch_size
        self.policy_id = policy_id

    async def rerank(
        self,
        query: str,
        candidates: tuple[MemoryHit, ...],
        limit: int,
        /,
    ) -> MemoryRerankDecision:
        """Score every candidate once and return the stable global selection."""

        _validate_bounds(len(candidates), limit)
        scores: dict[int, float] = {}
        for start in range(0, len(candidates), self._batch_size):
            batch = tuple(enumerate(candidates[start : start + self._batch_size], start=start + 1))
            response = await self._client.post(
                self._endpoint,
                json={
                    "query": query,
                    "candidates": [
                        {"rank": rank, "text": candidate.text} for rank, candidate in batch
                    ],
                },
            )
            response.raise_for_status()
            for rank, score in _parse_scores(response, expected_ranks=frozenset(rank for rank, _ in batch)):
                scores[rank] = score
        selected_ranks = tuple(
            rank
            for rank, _ in sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:limit]
        )
        return MemoryRerankDecision(
            selected_ranks=selected_ranks,
            usage=InferenceUsage(requests=_request_count(len(candidates), self._batch_size)),
        )


def _request_count(candidate_count: int, batch_size: int) -> int:
    """Number of HTTP requests a complete run over the pool issues."""

    return (candidate_count + batch_size - 1) // batch_size


def _validate_bounds(candidate_count: int, limit: int) -> None:
    if candidate_count < 1:
        raise ValueError("rerank requires at least one candidate")  # noqa: TRY003
    if limit < 1 or limit > candidate_count:
        raise ValueError("rerank limit must be between one and the candidate count")  # noqa: TRY003


def _parse_scores(
    response: httpx.Response,
    *,
    expected_ranks: frozenset[int],
) -> Sequence[tuple[int, float]]:
    """Validate one score response against the ranks this batch sent.

    The payload must be a JSON object with exactly one ``scores`` key whose
    value is a list of records with exactly ``rank`` and ``score`` fields.
    Records may arrive in any order but must cover this batch's ranks exactly
    once — no missing, duplicate, or foreign ranks — and every score must be a
    finite number.
    """

    try:
        payload: Any = json.loads(response.content)
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ValueError("rerank response is not valid JSON") from error  # noqa: TRY003
    if not isinstance(payload, dict) or set(payload) != {"scores"}:
        raise ValueError("rerank response must contain exactly a 'scores' field")  # noqa: TRY003
    score_records = payload["scores"]
    if not isinstance(score_records, list):
        raise ValueError("rerank 'scores' must be a list")  # noqa: TRY003
    parsed: list[tuple[int, float]] = []
    seen_ranks: set[int] = set()
    for score_record in score_records:
        if not isinstance(score_record, dict) or set(score_record) != _SCORE_RECORD_KEYS:
            raise ValueError("each score record must contain exactly 'rank' and 'score'")  # noqa: TRY003
        rank = score_record["rank"]
        score = score_record["score"]
        if type(rank) is not int:
            raise ValueError("score record 'rank' must be an integer")  # noqa: TRY003
        if rank in seen_ranks or rank not in expected_ranks:
            raise ValueError(f"score record rank {rank} is not part of this batch exactly once")  # noqa: TRY003
        if type(score) is not int and type(score) is not float:
            raise ValueError("score record 'score' must be a number")  # noqa: TRY003
        if not math.isfinite(score):
            raise ValueError("score record 'score' must be finite")  # noqa: TRY003
        seen_ranks.add(rank)
        parsed.append((rank, score))
    if seen_ranks != expected_ranks:
        raise ValueError("rerank scores must cover every rank of the batch exactly once")  # noqa: TRY003
    return parsed


__all__ = [
    "DEFAULT_HTTP_RERANK_BATCH_SIZE",
    "DEFAULT_HTTP_RERANK_POLICY_ID",
    "HTTPMemoryReranker",
]
