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

"""Query-driven selection of literal evidence fragments from Memory candidates.

The evidence-selection mode of :class:`PreparedContextBuilder` picks verbatim
fragments of the recalled Memory entries that match the current query instead
of truncating whole entries from the front. Fragments keep their exact
character coordinates so every piece stays verifiable against the cited entry
version, and the full rendered output (trust policy, envelope, citations and
content) is packed under the caller's UTF-8 byte budget.

Selection is purely lexical and offline: sentences are scored against the
query with IDF-weighted term overlap over the candidate pool, then packed by
value-per-byte so that extending an already-selected region (which costs no
extra citation overhead) competes fairly with starting a new fragment.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

from powercontext.artifacts import ArtifactAddress, ArtifactRef
from powercontext.builtin.runtime.models import PrepareContextRequest, PreparedContext
from powercontext.builtin.runtime.prepared_context import (
    _BEGIN_MARKER,
    _END_MARKER,
    MemoryEntryAddress,
    PreparedContextBuild,
    PreparedContextFragment,
    PreparedContextOmissions,
    PreparedMemoryCandidates,
)
from powercontext.builtin.runtime.prepared_text import TRUST_POLICY

_MAX_SENTENCE_BYTES = 600
_CHUNK_BYTES = 380
_RANK_DECAY = 0.05
_MAX_TERM_FREQUENCY = 3
_MAX_QUERY_FREQUENCY = 3

_SENTENCE_BOUNDARY = re.compile(
    r"[.!?\u2026\u3002\uff01\uff1f]+[\"'\u201d\u2019)\]\}\u300d\u300b]*(?:\s+|$)"
    r"|\n+"
)
_WORD_RUN = re.compile(r"\S+\s*")
_WORD_TOKEN = re.compile(r"[a-z]+|\d+")
# CJK ideographs, kana, Hangul syllables and compatibility ideographs have no
# word boundaries, so each character is treated as one token.
_CJK_RANGES = (("\u3040", "\u9fff"), ("\uac00", "\ud7af"), ("\uf900", "\ufaff"))

_EMPTY_ITEMS_JSON = json.dumps(
    {"trust": "untrusted_history", "items": []}, ensure_ascii=False, separators=(",", ":")
)
_JSON_PREFIX = _EMPTY_ITEMS_JSON[:-2]
_JSON_SUFFIX = "]}"
_FIXED_BYTES = len(
    (TRUST_POLICY + "\n\n" + _BEGIN_MARKER + "\n" + _JSON_PREFIX + _JSON_SUFFIX + "\n" + _END_MARKER).encode("utf-8")
)


@dataclass(frozen=True)
class _EvidenceEntry:
    """One deduplicated Memory candidate with its exact identity."""

    scope_id: str
    memory_ref: ArtifactRef
    entry_id: str
    entry_version_id: str
    text: str


@dataclass
class _Candidate:
    """One selectable span (sentence or chunk of an over-long sentence)."""

    entry_index: int
    span_index: int
    start: int
    end: int
    escaped_bytes: int
    tokens: Counter[str]
    score: float = 0.0


@dataclass(frozen=True)
class _RunPlan:
    """One merged fragment: consecutive selected spans of a single entry."""

    entry_index: int
    start: int
    end: int
    score: float


def _token_counts(text: str) -> Counter[str]:
    counts: Counter[str] = Counter()
    lowered = text.lower()
    for match in _WORD_TOKEN.finditer(lowered):
        counts[match.group()] += 1
    for char in lowered:
        for low, high in _CJK_RANGES:
            if low <= char <= high:
                counts[char] += 1
                break
    return counts


def _escaped_bytes(text: str) -> int:
    """UTF-8 bytes ``text`` occupies inside a JSON string literal.

    ``json.dumps`` escapes character by character, so this measure is additive
    when adjacent spans are merged into one fragment.
    """

    return len(json.dumps(text, ensure_ascii=False).encode("utf-8")) - 2


def _sentence_spans(text: str) -> tuple[tuple[int, int], ...]:
    """Partition ``text`` into contiguous sentence spans (including separators)."""

    spans: list[tuple[int, int]] = []
    start = 0
    for match in _SENTENCE_BOUNDARY.finditer(text):
        end = match.end()
        if end > start:
            spans.append((start, end))
            start = end
    if start < len(text):
        spans.append((start, len(text)))
    if not spans:
        return ((0, len(text)),) if text else ()
    return tuple(spans)


def _split_span(text: str, start: int, end: int) -> list[tuple[int, int]]:
    """Split an over-long span at word boundaries into byte-bounded chunks."""

    pieces: list[tuple[int, int]] = []
    piece_start = start
    piece_bytes = 0
    last_break = start
    for match in _WORD_RUN.finditer(text, start, end):
        word_bytes = len(match.group().encode("utf-8"))
        if piece_bytes and piece_bytes + word_bytes > _CHUNK_BYTES:
            pieces.append((piece_start, last_break))
            piece_start = last_break
            piece_bytes = 0
        piece_bytes += word_bytes
        last_break = match.end()
    pieces.append((piece_start, end))
    return pieces


def _candidate_spans(text: str) -> tuple[tuple[int, int], ...]:
    spans: list[tuple[int, int]] = []
    for start, end in _sentence_spans(text):
        if len(text[start : end].encode("utf-8")) <= _MAX_SENTENCE_BYTES:
            spans.append((start, end))
        else:
            spans.extend(_split_span(text, start, end))
    return tuple(spans)


def _collect_entries(
    current_scope_id: str | None,
    memory_candidates: Sequence[PreparedMemoryCandidates],
) -> tuple[_EvidenceEntry, ...]:
    entries: list[_EvidenceEntry] = []
    seen: set[tuple[str, str, str]] = set()
    for group in memory_candidates:
        scope_id = group.scope_id or (current_scope_id or "")
        if not scope_id:
            continue
        for hit in group.hits:
            key = (scope_id, hit.entry_id, hit.entry_version_id)
            if key in seen:
                continue
            seen.add(key)
            if not hit.text.strip():
                continue
            entries.append(
                _EvidenceEntry(
                    scope_id=scope_id,
                    memory_ref=hit.memory_ref,
                    entry_id=hit.entry_id,
                    entry_version_id=hit.entry_version_id,
                    text=hit.text,
                )
            )
    return tuple(entries)


def _score_candidates(entries: Sequence[_EvidenceEntry], query: str) -> tuple[list[_Candidate], ...]:
    """Build per-entry span candidates and score them against the query.

    Sentences keep their order inside each entry so that consecutive span
    indices are textually adjacent and can merge back into one fragment.
    """

    per_entry: list[list[_Candidate]] = []
    for entry_index, entry in enumerate(entries):
        candidates: list[_Candidate] = []
        for span_index, (start, end) in enumerate(_candidate_spans(entry.text)):
            segment = entry.text[start:end]
            candidates.append(
                _Candidate(
                    entry_index=entry_index,
                    span_index=span_index,
                    start=start,
                    end=end,
                    escaped_bytes=_escaped_bytes(segment),
                    tokens=_token_counts(segment),
                )
            )
        per_entry.append(candidates)

    total = sum(len(group) for group in per_entry)
    if total == 0:
        return tuple(per_entry)

    document_frequency: Counter[str] = Counter()
    for group in per_entry:
        for candidate in group:
            document_frequency.update(candidate.tokens.keys())
    query_counts = _token_counts(query)
    weights: dict[str, float] = {}
    for token, frequency in query_counts.items():
        hits = document_frequency[token]
        if hits:
            idf = math.log(1.0 + total / hits)
            weights[token] = idf * (1.0 + 0.5 * (min(frequency, _MAX_QUERY_FREQUENCY) - 1))
    if not weights:
        return tuple(per_entry)

    for entry_index, group in enumerate(per_entry):
        decay = 1.0 / (1.0 + _RANK_DECAY * entry_index)
        for candidate in group:
            score = 0.0
            for token, weight in weights.items():
                frequency = candidate.tokens.get(token)
                if frequency:
                    score += weight * min(frequency, _MAX_TERM_FREQUENCY)
            candidate.score = score * decay
    return tuple(per_entry)


class _Packer:
    """Greedy byte-budget packer with exact incremental cost tracking.

    Candidates merge with already-selected neighbours for free (one citation
    fewer to pay), so the packer prefers value-per-byte: extending a strong
    region competes fairly with starting a new fragment elsewhere.
    """

    def __init__(self, entries: Sequence[_EvidenceEntry], per_entry: Sequence[Sequence[_Candidate]]) -> None:
        self._entries = entries
        self._per_entry = per_entry
        self._selected: list[set[int]] = [set() for _ in entries]
        self._citation_bytes = [self._citation_size(entry) for entry in entries]
        self.total = _FIXED_BYTES
        self.selected_count = 0

    @staticmethod
    def _citation_size(entry: _EvidenceEntry) -> int:
        citation = {
            "scope_id": entry.scope_id,
            "memory_ref": entry.memory_ref.model_dump(mode="json"),
            "entry_id": entry.entry_id,
            "entry_version_id": entry.entry_version_id,
        }
        return len(json.dumps(citation, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))

    def _item_base(self, entry_index: int) -> int:
        return 40 + self._citation_bytes[entry_index]

    def _runs(self, entry_index: int) -> list[tuple[int, int]]:
        runs: list[tuple[int, int]] = []
        for index in sorted(self._selected[entry_index]):
            if runs and index == runs[-1][1] + 1:
                runs[-1] = (runs[-1][0], index)
            else:
                runs.append((index, index))
        return runs

    def _run_containing(self, entry_index: int, span_index: int) -> tuple[int, int]:
        for run in self._runs(entry_index):
            if run[0] <= span_index <= run[1]:
                return run
        raise KeyError(span_index)  # pragma: no cover - caller guarantees membership

    def incremental_cost(self, candidate: _Candidate) -> int:
        entry_index = candidate.entry_index
        selected = self._selected[entry_index]
        spans = self._per_entry[entry_index]
        index = candidate.span_index
        has_previous = index - 1 in selected
        has_next = index + 1 in selected
        base = self._item_base(entry_index)
        if not has_previous and not has_next:
            comma = 1 if self.selected_count else 0
            return (
                base
                + len(str(candidate.start))
                + len(str(candidate.end))
                + candidate.escaped_bytes
                + comma
            )
        if has_previous and has_next:
            left_end = spans[self._run_containing(entry_index, index - 1)[1]].end
            right_start = spans[self._run_containing(entry_index, index + 1)[0]].start
            return candidate.escaped_bytes - base - len(str(left_end)) - len(str(right_start)) - 1
        if has_previous:
            left_end = spans[self._run_containing(entry_index, index - 1)[1]].end
            return candidate.escaped_bytes + len(str(candidate.end)) - len(str(left_end))
        right_start = spans[self._run_containing(entry_index, index + 1)[0]].start
        return candidate.escaped_bytes + len(str(candidate.start)) - len(str(right_start))

    def add(self, candidate: _Candidate) -> None:
        self.total += self.incremental_cost(candidate)
        self._selected[candidate.entry_index].add(candidate.span_index)
        self.selected_count += 1

    def is_selected(self, candidate: _Candidate) -> bool:
        return candidate.span_index in self._selected[candidate.entry_index]

    def selection(self) -> set[tuple[int, int]]:
        return {
            (entry_index, span_index)
            for entry_index, spans in enumerate(self._selected)
            for span_index in spans
        }


def _pack(per_entry: Sequence[Sequence[_Candidate]], entries: Sequence[_EvidenceEntry], max_bytes: int) -> _Packer:
    packer = _Packer(entries, per_entry)
    all_candidates = [candidate for group in per_entry for candidate in group]
    while True:
        best_key: tuple | None = None
        best_candidate: _Candidate | None = None
        for candidate in all_candidates:
            if packer.is_selected(candidate):
                continue
            cost = packer.incremental_cost(candidate)
            if cost > 0 and packer.total + cost > max_bytes:
                continue
            if cost <= 0:
                # Merging for free (or even freeing bytes): always attractive.
                key = (0, 0.0, 0, cost, candidate.entry_index, candidate.span_index)
            elif candidate.score > 0.0:
                key = (0, -candidate.score / cost, 0, cost, candidate.entry_index, candidate.span_index)
            else:
                selected = packer._selected[candidate.entry_index]
                adjacent = candidate.span_index - 1 in selected or candidate.span_index + 1 in selected
                key = (1, 0.0, 0 if adjacent else 1, cost, candidate.entry_index, candidate.span_index)
            if best_key is None or key < best_key:
                best_key = key
                best_candidate = candidate
        if best_candidate is None:
            break
        packer.add(best_candidate)
    return packer


def _run_plans(
    per_entry: Sequence[Sequence[_Candidate]],
    selected: set[tuple[int, int]],
) -> list[_RunPlan]:
    plans: list[_RunPlan] = []
    for entry_index, spans in enumerate(per_entry):
        indices = sorted(span_index for entry, span_index in selected if entry == entry_index)
        run: list[int] = []
        for index in indices:
            if run and index == run[-1] + 1:
                run.append(index)
            else:
                if run:
                    plans.append(_plan(entry_index, spans, run))
                run = [index]
        if run:
            plans.append(_plan(entry_index, spans, run))
    plans.sort(key=lambda plan: (plan.entry_index, plan.start))
    return plans


def _plan(entry_index: int, spans: Sequence[_Candidate], run: Sequence[int]) -> _RunPlan:
    return _RunPlan(
        entry_index=entry_index,
        start=spans[run[0]].start,
        end=spans[run[-1]].end,
        score=sum(spans[index].score for index in run),
    )


def _render_content(items: Sequence[dict[str, object]]) -> str:
    envelope = {"trust": "untrusted_history", "items": list(items)}
    encoded = json.dumps(envelope, ensure_ascii=False, separators=(",", ":"))
    return "\n\n".join((TRUST_POLICY, f"{_BEGIN_MARKER}\n{encoded}\n{_END_MARKER}"))


def _item(entry: _EvidenceEntry, start: int, end: int) -> dict[str, object]:
    return {
        "citation": {
            "scope_id": entry.scope_id,
            "memory_ref": entry.memory_ref.model_dump(mode="json"),
            "entry_id": entry.entry_id,
            "entry_version_id": entry.entry_version_id,
        },
        "content": entry.text[start:end],
        "start": start,
        "end": end,
    }


def _assemble(
    entries: Sequence[_EvidenceEntry],
    per_entry: Sequence[Sequence[_Candidate]],
    selected: set[tuple[int, int]],
    max_bytes: int,
) -> tuple[str | None, list[_RunPlan]]:
    """Render the final content, trimming under-scoring boundary spans if needed.

    The packer's incremental accounting is exact, so the trim loop is a safety
    net rather than the primary budget mechanism: it removes the lowest-score
    run-boundary candidate until the real rendered size fits.
    """

    while selected:
        plans = _run_plans(per_entry, selected)
        content = _render_content([_item(entries[plan.entry_index], plan.start, plan.end) for plan in plans])
        if len(content.encode("utf-8")) <= max_bytes:
            return content, plans
        boundary = [
            (entry_index, span_index)
            for entry_index, span_index in selected
            if (entry_index, span_index + 1) not in selected or (entry_index, span_index - 1) not in selected
        ]
        if not boundary:
            selected.clear()
            break
        victim = min(
            boundary,
            key=lambda key: (per_entry[key[0]][key[1]].score, key[0], key[1]),
        )
        selected.discard(victim)
    return None, []


def build_evidence_result(
    *,
    request: PrepareContextRequest,
    current_scope_id: str | None,
    memory_candidates: Sequence[PreparedMemoryCandidates],
) -> PreparedContextBuild:
    """Select query-matched literal fragments and pack them under the budget."""

    entries = _collect_entries(current_scope_id, memory_candidates)
    per_entry = _score_candidates(entries, request.query)
    candidate_count = sum(len(group) for group in per_entry)
    omissions = PreparedContextOmissions()
    if not entries or candidate_count == 0:
        return PreparedContextBuild(context=_empty_context(), origins=(), omissions=omissions, fragments=())

    packer = _pack(per_entry, entries, request.max_bytes)
    content, plans = _assemble(entries, per_entry, packer.selection(), request.max_bytes)
    if content is None or not plans:
        return PreparedContextBuild(
            context=_empty_context(),
            origins=(),
            omissions=PreparedContextOmissions(dropped_items=candidate_count),
            fragments=(),
        )

    fragments: list[PreparedContextFragment] = []
    origins: list[MemoryEntryAddress] = []
    items: list[dict[str, object]] = []
    for plan in plans:
        entry = entries[plan.entry_index]
        address = MemoryEntryAddress(
            memory=ArtifactAddress(scope_id=entry.scope_id, artifact=entry.memory_ref),
            entry_id=entry.entry_id,
            entry_version_id=entry.entry_version_id,
        )
        fragments.append(
            PreparedContextFragment(address=address, start=plan.start, end=plan.end, text=entry.text[plan.start : plan.end])
        )
        origins.append(address)
        items.append(_item(entry, plan.start, plan.end))
    omissions = PreparedContextOmissions(dropped_items=candidate_count - len(fragments))
    return PreparedContextBuild(
        context=PreparedContext(status="ready", content=content, content_bytes=len(content.encode("utf-8"))),
        origins=tuple(origins),
        omissions=omissions,
        fragments=tuple(fragments),
    )


def _empty_context() -> PreparedContext:
    return PreparedContext(status="empty", content=None, content_bytes=0)
