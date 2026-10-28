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

"""Context Pack wire protocol for incremental context transport.

A :class:`ContextPack` represents a full prepared context as an ordered tuple
of byte chunks plus a digest over the chunk sequence and a self-referential
``byte_count`` (the length of its own canonical serialization, including the
bytes occupied by the ``byte_count`` field itself).  A
:class:`ContextPackDelta` carries a retained prefix length of an already
delivered base pack plus the appended suffix chunks needed to rebuild the
target pack.  Receivers rebuild from serialized bytes only; no sender-side
state participates in decoding.

Wire contract (interoperable, oracle-checked):

- Pack JSON has exactly the fields ``schema`` (``powercontext.context-pack.v1``),
  ``items`` (standard Base64 string array), ``digest`` and ``byte_count``.
- ``digest`` is ``sha256:<hex>`` over the canonical JSON of an *unsigned*
  object holding only ``schema`` and ``items``.
- ``byte_count == len(pack.to_bytes())`` including its own digits; resolved by
  iteration to a fixed point.
- Delta JSON has exactly the fields ``schema``
  (``powercontext.context-pack-delta.v1``), ``base_digest``, ``target_digest``,
  ``prefix_items`` and ``appended``; ``delta.byte_count == len(delta.to_bytes())``
  is reported as a property and never written into the JSON.
- JSON encoding is UTF-8 with sorted keys and ``,``/``:`` separators (no
  extra whitespace).
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import re
from dataclasses import dataclass, field

CONTEXT_PACK_SCHEMA = "powercontext.context-pack.v1"
CONTEXT_PACK_DELTA_SCHEMA = "powercontext.context-pack-delta.v1"
DEFAULT_CHUNK_BYTES = 512

_PACK_FIELDS = frozenset({"schema", "items", "digest", "byte_count"})
_DELTA_FIELDS = frozenset({"schema", "base_digest", "target_digest", "prefix_items", "appended"})
_DIGEST_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")
_MAX_BYTE_COUNT_ITERATIONS = 8


class ContextPackError(ValueError):
    """Raised when pack construction, chunking, or wire decoding violates the protocol."""


def _canonical_json(payload: dict[str, object]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _decode_base64_items(value: object, *, field_name: str) -> tuple[bytes, ...]:
    if not isinstance(value, list):
        raise ContextPackError(f"{field_name} must be a JSON array of Base64 strings")
    items: list[bytes] = []
    for entry in value:
        if not isinstance(entry, str):
            raise ContextPackError(f"{field_name} entries must be Base64 strings")
        try:
            items.append(base64.b64decode(entry, validate=True))
        except (binascii.Error, ValueError) as error:
            raise ContextPackError(f"{field_name} contains invalid Base64: {error}") from error
    return tuple(items)


def _encode_base64_items(items: tuple[bytes, ...], *, field_name: str) -> list[str]:
    encoded: list[str] = []
    for item in items:
        if not isinstance(item, (bytes, bytearray, memoryview)):
            raise ContextPackError(f"{field_name} entries must be bytes-like")
        encoded.append(base64.b64encode(bytes(item)).decode("ascii"))
    return encoded


def _compute_digest(items: tuple[bytes, ...]) -> str:
    unsigned = _canonical_json({
        "schema": CONTEXT_PACK_SCHEMA,
        "items": _encode_base64_items(items, field_name="items"),
    })
    return "sha256:" + hashlib.sha256(unsigned).hexdigest()


def _validate_digest(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not _DIGEST_PATTERN.match(value):
        raise ContextPackError(f"{field_name} must be a 'sha256:<64 hex digits>' string")
    return value


@dataclass(frozen=True)
class ContextPack:
    """A complete context pack: chunked bytes, a sequence digest, and wire bytes."""

    items: tuple[bytes, ...]
    _digest: str = field(repr=False, default="")
    _byte_count: int = field(repr=False, default=0)

    def __post_init__(self) -> None:
        normalized: list[bytes] = []
        for item in self.items:
            if not isinstance(item, (bytes, bytearray, memoryview)):
                raise ContextPackError("pack items must be bytes-like")
            normalized.append(bytes(item))
        object.__setattr__(self, "items", tuple(normalized))
        object.__setattr__(self, "_digest", _compute_digest(self.items))
        object.__setattr__(self, "_byte_count", self._resolve_byte_count())

    @classmethod
    def build(cls, items: tuple[bytes, ...]) -> ContextPack:
        """Build a pack from an ordered tuple of byte chunks."""
        if isinstance(items, (bytes, bytearray, memoryview)) or not isinstance(items, (tuple, list)):
            raise ContextPackError("items must be a tuple of byte chunks")
        return cls(tuple(items))

    @classmethod
    def from_content(cls, content: str, *, chunk_bytes: int = DEFAULT_CHUNK_BYTES) -> ContextPack:
        """Deterministically chunk ``content`` into bounded, valid UTF-8 chunks.

        Chunking splits on line boundaries first; lines longer than the cap are
        split at UTF-8 character boundaries.  Reconstruction by concatenation
        always yields the original bytes.
        """
        return cls(_chunk_content(content, chunk_bytes))

    def _serialize(self, byte_count: int) -> bytes:
        return _canonical_json({
            "schema": CONTEXT_PACK_SCHEMA,
            "items": _encode_base64_items(self.items, field_name="items"),
            "digest": self._digest,
            "byte_count": byte_count,
        })

    def _resolve_byte_count(self) -> int:
        # ``byte_count`` covers its own digits, so iterate to the fixed point.
        guess = len(_canonical_json({
            "schema": CONTEXT_PACK_SCHEMA,
            "items": _encode_base64_items(self.items, field_name="items"),
        }))
        for _ in range(_MAX_BYTE_COUNT_ITERATIONS):
            data = self._serialize(guess)
            if len(data) == guess:
                return guess
            guess = len(data)
        raise ContextPackError("byte_count failed to converge")  # pragma: no cover - unreachable

    @property
    def digest(self) -> str:
        return self._digest

    @property
    def byte_count(self) -> int:
        return self._byte_count

    def to_bytes(self) -> bytes:
        """Canonical wire serialization (sorted keys, compact separators, UTF-8)."""
        return self._serialize(self._byte_count)

    @classmethod
    def from_bytes(cls, data: bytes) -> ContextPack:
        """Decode and fully validate a serialized pack (digest and byte count)."""
        payload = _parse_wire_object(data, _PACK_FIELDS, "context pack")
        if payload["schema"] != CONTEXT_PACK_SCHEMA:
            raise ContextPackError("unexpected context pack schema")
        items = _decode_base64_items(payload["items"], field_name="items")
        rebuilt = cls(items)
        if payload["digest"] != rebuilt._digest:
            raise ContextPackError("pack digest mismatch")
        byte_count = payload["byte_count"]
        if type(byte_count) is not int:
            raise ContextPackError("byte_count must be an integer")
        if byte_count != rebuilt.byte_count:
            raise ContextPackError("pack byte_count mismatch")
        return rebuilt


def _parse_wire_object(data: bytes, expected_fields: frozenset[str], label: str) -> dict[str, object]:
    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise ContextPackError(f"{label} payload must be bytes")
    try:
        payload = json.loads(bytes(data).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ContextPackError(f"{label} payload is not valid UTF-8 JSON: {error}") from error
    if not isinstance(payload, dict):
        raise ContextPackError(f"{label} payload must be a JSON object")
    if set(payload) != expected_fields:
        raise ContextPackError(f"{label} fields must be exactly {sorted(expected_fields)}")
    return payload


def _split_at_utf8_boundary(data: bytes, cap: int) -> tuple[bytes, ...]:
    pieces: list[bytes] = []
    position = 0
    total = len(data)
    while position < total:
        end = min(position + cap, total)
        while end > position:
            try:
                data[position:end].decode("utf-8")
                break
            except UnicodeDecodeError:
                end -= 1
        if end == position:
            raise ContextPackError("chunk_bytes is too small to hold one UTF-8 character")
        pieces.append(data[position:end])
        position = end
    return tuple(pieces)


def _chunk_content(content: str, chunk_bytes: int) -> tuple[bytes, ...]:
    if type(chunk_bytes) is not int or chunk_bytes <= 0:
        raise ContextPackError("chunk_bytes must be a positive integer")
    try:
        data = content.encode("utf-8")
    except UnicodeEncodeError as error:
        raise ContextPackError("content is not encodable as UTF-8") from error
    if not data:
        return ()
    items: list[bytes] = []
    for line in content.splitlines(keepends=True):
        line_bytes = line.encode("utf-8")
        if len(line_bytes) <= chunk_bytes:
            items.append(line_bytes)
        else:
            items.extend(_split_at_utf8_boundary(line_bytes, chunk_bytes))
    return tuple(items)


@dataclass(frozen=True)
class ContextPackDelta:
    """An incremental transmission: keep ``prefix_items`` base chunks, append the rest."""

    base_digest: str
    target_digest: str
    prefix_items: int
    appended: tuple[bytes, ...]
    _byte_count: int = field(repr=False, default=0)

    def __post_init__(self) -> None:
        appended: list[bytes] = []
        for item in self.appended:
            if not isinstance(item, (bytes, bytearray, memoryview)):
                raise ContextPackError("delta appended entries must be bytes-like")
            appended.append(bytes(item))
        object.__setattr__(self, "appended", tuple(appended))
        object.__setattr__(self, "_byte_count", len(self._serialize()))

    @classmethod
    def between(cls, base: ContextPack, target: ContextPack) -> ContextPackDelta:
        """Choose the longest common chunk prefix of ``base`` and ``target``."""
        if not isinstance(base, ContextPack) or not isinstance(target, ContextPack):
            raise ContextPackError("delta endpoints must be ContextPack instances")
        prefix = 0
        for base_item, target_item in zip(base.items, target.items):
            if base_item != target_item:
                break
            prefix += 1
        return cls(
            base_digest=base.digest,
            target_digest=target.digest,
            prefix_items=prefix,
            appended=target.items[prefix:],
        )

    def _serialize(self) -> bytes:
        return _canonical_json({
            "schema": CONTEXT_PACK_DELTA_SCHEMA,
            "base_digest": self.base_digest,
            "target_digest": self.target_digest,
            "prefix_items": self.prefix_items,
            "appended": _encode_base64_items(self.appended, field_name="appended"),
        })

    @property
    def byte_count(self) -> int:
        return self._byte_count

    def to_bytes(self) -> bytes:
        """Canonical wire serialization (``byte_count`` is not part of the JSON)."""
        return self._serialize()

    @classmethod
    def from_bytes(cls, data: bytes) -> ContextPackDelta:
        """Decode and validate a serialized delta (structural and digest format checks)."""
        payload = _parse_wire_object(data, _DELTA_FIELDS, "context pack delta")
        if payload["schema"] != CONTEXT_PACK_DELTA_SCHEMA:
            raise ContextPackError("unexpected context pack delta schema")
        base_digest = _validate_digest(payload["base_digest"], field_name="base_digest")
        target_digest = _validate_digest(payload["target_digest"], field_name="target_digest")
        prefix_items = payload["prefix_items"]
        if type(prefix_items) is not int:
            raise ContextPackError("prefix_items must be an integer")
        if prefix_items < 0:
            raise ContextPackError("prefix_items must be non-negative")
        appended = _decode_base64_items(payload["appended"], field_name="appended")
        return cls(
            base_digest=base_digest,
            target_digest=target_digest,
            prefix_items=prefix_items,
            appended=appended,
        )


def apply_context_pack_delta(base: ContextPack, delta: ContextPackDelta) -> ContextPack:
    """Validate ``delta`` against ``base`` and rebuild the target pack."""
    if not isinstance(base, ContextPack) or not isinstance(delta, ContextPackDelta):
        raise ContextPackError("apply_context_pack_delta requires a pack and a delta")
    if delta.base_digest != base.digest:
        raise ContextPackError("delta base digest does not match the provided base pack")
    prefix = delta.prefix_items
    if type(prefix) is not int:
        raise ContextPackError("prefix_items must be an integer")
    if prefix < 0 or prefix > len(base.items):
        raise ContextPackError("prefix_items is out of bounds for the base pack")
    rebuilt = ContextPack.build((*base.items[:prefix], *delta.appended))
    if rebuilt.digest != delta.target_digest:
        raise ContextPackError("rebuilt pack digest does not match the delta target digest")
    return rebuilt
