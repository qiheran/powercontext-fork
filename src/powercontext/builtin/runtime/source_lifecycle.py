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

"""Source lifecycle: propagate evidence withdrawal and restore to derivations.

Registrations form a per-scope DAG whose roots are captured Sources. Memory
entries and Topic Memory artifacts join the graph with an ``all``/``any``
support policy over their parents. Withdrawal and restore events on a Source
recompute every descendant's availability, deactivate unsupported Memory
entries through the native Memory state transition (so list, search and
Context all reflect the result without rewriting any body version), and keep
unsupported Topic Memory artifacts out of native Topic search and Context
while their exact ArtifactRef stays readable.

State is persisted per scope in one row of ``pc_source_lifecycle`` so
registrations, events and availability survive restarts. A native Memory
write that fails mid-propagation is healed by replaying the same event (or
any later lifecycle call): the stored availability is the source of truth and
every apply re-converges the native states idempotently.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictInt, field_validator
from sqlalchemy import delete, insert, select

from powercontext.artifacts import ArtifactRef, MemoryCitation
from powercontext.builtin.artifacts.memory import Memory
from powercontext.builtin.persistence.tables import SOURCE_LIFECYCLE_TABLE
from powercontext.builtin.sources import validate_scope_id
from powercontext.sources import SourceRef

if TYPE_CHECKING:
    from powercontext.builtin.artifacts.memory import MemoryService
    from powercontext.builtin.runtime.application import BuiltinRuntime

LIFECYCLE_STATE_SCHEMA = "powercontext.source-lifecycle-state.v1"
SOURCE_NODE_KIND = "source"
MEMORY_NODE_KIND = "memory"
TOPIC_KIND = "topic"
TOPIC_MEMORY_FAMILY = "topic-memory"
MEMORY_FAMILY = "memory"
SUPPORT_POLICIES = ("all", "any")
RESOLVE_PURPOSES = ("generation", "audit")
MAX_NODE_ID_LENGTH = 256


class SourceLifecycleError(ValueError):
    """Reject invalid lifecycle registrations, events and resolutions."""


class SourceLifecycleEvent(BaseModel):
    """One withdrawal or restore event applied to a registered Source."""

    model_config = ConfigDict(extra="forbid")

    node_id: str
    reason: str
    revision: StrictInt = Field(gt=0)
    sequence: StrictInt = Field(gt=0)
    available: StrictBool

    @field_validator("node_id", "reason")
    @classmethod
    def _require_visible_text(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("must be non-empty and not only whitespace")  # noqa: TRY003
        return value


class _LifecycleEventRecord(BaseModel):
    """The last accepted event for one Source, kept for idempotent replay."""

    sequence: int
    available: bool
    reason: str


class LifecycleNode(BaseModel):
    """One registered DAG node with its binding and current availability."""

    node_id: str
    kind: Literal["source", "memory", "topic"]
    ref: dict[str, Any]
    revision: int | None = None
    parents: tuple[str, ...] = ()
    policy: str | None = None
    available: bool = True


class LifecycleState(BaseModel):
    """The persisted lifecycle graph of one scope."""

    schema_: str = LIFECYCLE_STATE_SCHEMA
    nodes: list[LifecycleNode] = []
    events: dict[str, _LifecycleEventRecord] = {}

    def node(self, node_id: str) -> LifecycleNode | None:
        return next((candidate for candidate in self.nodes if candidate.node_id == node_id), None)

    def active_ids(self) -> list[str]:
        return sorted(node.node_id for node in self.nodes if node.available)


class SourceLifecycleApplication:
    """Select the scoped Source lifecycle service."""

    def __init__(self, runtime: BuiltinRuntime) -> None:
        self._runtime = runtime

    def for_scope(self, scope_id: str, /) -> ScopedSourceLifecycleApplication:
        return ScopedSourceLifecycleApplication(self._runtime, scope_id)


class ScopedSourceLifecycleApplication:
    """Register the evidence DAG and propagate withdrawal/restore in one scope."""

    def __init__(self, runtime: BuiltinRuntime, scope_id: str) -> None:
        self._runtime = runtime
        self.scope_id = validate_scope_id(scope_id)

    async def register_source(
        self,
        node_id: str,
        source_ref: SourceRef,
        /,
        *,
        revision: int = 3,
    ) -> None:
        cleaned = _clean_node_id(node_id)
        if isinstance(revision, bool) or not isinstance(revision, int) or revision < 1:
            raise SourceLifecycleError("lifecycle-revision")
        if not isinstance(source_ref, SourceRef):
            raise SourceLifecycleError("source-ref")
        ref = {"source_type": source_ref.source_type, "source_id": source_ref.source_id}
        async with self._runtime._scope_operation(self.scope_id), self._runtime._locked(self.scope_id):
            state = await self._load_state()
            existing = state.node(cleaned)
            if existing is not None:
                _require_same_binding(existing, kind=SOURCE_NODE_KIND, ref=ref, parents=(), policy=None)
                return
            await self._require_source(source_ref)
            state.nodes.append(
                LifecycleNode(node_id=cleaned, kind=SOURCE_NODE_KIND, ref=ref, revision=revision, available=True)
            )
            await self._save_state(state)
            await self._sync_native_memory(state)

    async def register_memory(
        self,
        node_id: str,
        citation: MemoryCitation,
        /,
        *,
        parents: tuple[str, ...],
        policy: str,
    ) -> None:
        cleaned = _clean_node_id(node_id)
        cleaned_parents = _clean_parents(parents)
        _require_policy(policy)
        ref = _citation_ref(citation)
        async with self._runtime._scope_operation(self.scope_id), self._runtime._locked(self.scope_id):
            state = await self._load_state()
            existing = state.node(cleaned)
            if existing is not None:
                _require_same_binding(existing, kind=MEMORY_NODE_KIND, ref=ref, parents=cleaned_parents, policy=policy)
                return
            _require_registered_parents(state, cleaned, cleaned_parents)
            await self._require_memory_citation(citation)
            available = _supported(cleaned_parents, policy, state)
            state.nodes.append(
                LifecycleNode(
                    node_id=cleaned,
                    kind=MEMORY_NODE_KIND,
                    ref=ref,
                    parents=cleaned_parents,
                    policy=policy,
                    available=available,
                )
            )
            await self._save_state(state)
            await self._sync_native_memory(state)

    async def register_topic(
        self,
        node_id: str,
        artifact_ref: ArtifactRef,
        /,
        *,
        parents: tuple[str, ...],
        policy: str,
    ) -> None:
        cleaned = _clean_node_id(node_id)
        cleaned_parents = _clean_parents(parents)
        _require_policy(policy)
        ref = _artifact_ref_payload(artifact_ref)
        async with self._runtime._scope_operation(self.scope_id), self._runtime._locked(self.scope_id):
            state = await self._load_state()
            existing = state.node(cleaned)
            if existing is not None:
                _require_same_binding(existing, kind=TOPIC_KIND, ref=ref, parents=cleaned_parents, policy=policy)
                return
            _require_registered_parents(state, cleaned, cleaned_parents)
            await self._require_topic_artifact(artifact_ref)
            available = _supported(cleaned_parents, policy, state)
            state.nodes.append(
                LifecycleNode(
                    node_id=cleaned,
                    kind=TOPIC_KIND,
                    ref=ref,
                    parents=cleaned_parents,
                    policy=policy,
                    available=available,
                )
            )
            await self._save_state(state)
            await self._sync_native_memory(state)

    async def apply(self, event: SourceLifecycleEvent, /) -> list[str]:
        """Apply one withdrawal/restore event and return all available node ids."""

        if not isinstance(event, SourceLifecycleEvent):
            raise SourceLifecycleError("event-type")
        async with self._runtime._scope_operation(self.scope_id), self._runtime._locked(self.scope_id):
            state = await self._load_state()
            node = state.node(event.node_id)
            if node is None or node.kind != SOURCE_NODE_KIND:
                raise SourceLifecycleError("unknown-source")
            if node.revision != event.revision:
                raise SourceLifecycleError("revision-mismatch")
            last = state.events.get(event.node_id)
            if last is not None:
                if event.sequence < last.sequence:
                    raise SourceLifecycleError("sequence")
                if event.sequence == last.sequence:
                    replay = event.available == last.available and event.reason == last.reason
                    if not replay:
                        raise SourceLifecycleError("sequence")
                    # Idempotent replay: never recompute or rewrite, but finish any
                    # native propagation an earlier interrupted attempt left behind.
                    await self._sync_native_memory(state)
                    return state.active_ids()
            node.available = event.available
            state.events[event.node_id] = _LifecycleEventRecord(
                sequence=event.sequence,
                available=event.available,
                reason=event.reason,
            )
            _propagate(state)
            await self._save_state(state)
            await self._sync_native_memory(state)
            return state.active_ids()

    async def resolve(self, node_id: str, /, *, purpose: str) -> SourceRef | MemoryCitation | ArtifactRef:
        """Return the original registration reference for one purpose."""

        if purpose not in RESOLVE_PURPOSES:
            raise SourceLifecycleError("purpose")
        cleaned = _clean_node_id(node_id)
        async with self._runtime._scope_operation(self.scope_id):
            state = await self._load_state()
            node = state.node(cleaned)
            if node is None:
                raise SourceLifecycleError("unknown-node")
            if purpose == "generation" and not node.available:
                raise SourceLifecycleError("unavailable")
            return _reference(node)

    async def _load_state(self) -> LifecycleState:
        database = _database(self._runtime)
        async with database.transaction() as connection:
            row = (
                await connection.execute(
                    select(SOURCE_LIFECYCLE_TABLE.c.payload).where(
                        SOURCE_LIFECYCLE_TABLE.c.scope_id == self.scope_id
                    )
                )
            ).first()
        if row is None:
            return LifecycleState()
        return LifecycleState.model_validate(json.loads(bytes(row[0]).decode("utf-8")))

    async def _save_state(self, state: LifecycleState) -> None:
        payload = json.dumps(
            state.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        database = _database(self._runtime)
        async with database.transaction() as connection:
            await connection.execute(
                delete(SOURCE_LIFECYCLE_TABLE).where(SOURCE_LIFECYCLE_TABLE.c.scope_id == self.scope_id)
            )
            await connection.execute(
                insert(SOURCE_LIFECYCLE_TABLE).values(scope_id=self.scope_id, payload=payload)
            )

    async def _require_source(self, source_ref: SourceRef) -> None:
        from powercontext.builtin.records import BaseValueNotFoundError

        try:
            await self._runtime._records().get_source(self.scope_id, source_ref.source_type, source_ref.source_id)
        except BaseValueNotFoundError as error:
            raise SourceLifecycleError("source-ref") from error

    async def _require_memory_citation(self, citation: MemoryCitation) -> None:
        from powercontext.builtin.artifacts.memory.errors import (
            InvalidMemoryCitationError,
            MemoryEntryNotFoundError,
        )
        from powercontext.errors import ArtifactNotFoundError

        if citation.memory_ref.family != MEMORY_FAMILY:
            raise SourceLifecycleError("memory-family")
        try:
            async with self._runtime._context(self.scope_id) as context:
                service = context.artifacts.memory
                memory = await service.revision(citation.memory_ref)
                if memory.artifact_id != context.artifacts.memory_artifact_id:
                    raise SourceLifecycleError("memory-identity")
                if not any(
                    item.entry_id == citation.entry_id
                    and item.entry_version_id == citation.entry_version_id
                    for item in memory.content.manifest.entries
                ):
                    raise SourceLifecycleError("memory-entry")
                await service.validate_citation(citation)
        except (ArtifactNotFoundError, InvalidMemoryCitationError, MemoryEntryNotFoundError) as error:
            raise SourceLifecycleError("memory-citation") from error

    async def _require_topic_artifact(self, artifact_ref: ArtifactRef) -> None:
        from powercontext.builtin.records import BaseValueNotFoundError

        if artifact_ref.family != TOPIC_MEMORY_FAMILY:
            raise SourceLifecycleError("topic-family")
        try:
            await self._runtime._records().get_artifact_revision(
                self.scope_id,
                artifact_ref.family,
                artifact_ref.artifact_id,
                artifact_ref.revision,
            )
        except BaseValueNotFoundError as error:
            raise SourceLifecycleError("topic-ref") from error

    async def _sync_native_memory(self, state: LifecycleState) -> None:
        """Converge native Memory entry states with the persisted availability.

        Every call re-checks each registered Memory entry and only issues the
        native state transition when the current head disagrees, so replays and
        retrials after interrupted propagation never create extra revisions.
        """

        desired: dict[str, dict[str, bool]] = {}
        for node in state.nodes:
            if node.kind != MEMORY_NODE_KIND:
                continue
            entries = desired.setdefault(node.ref["memory_ref"]["artifact_id"], {})
            entries[node.ref["entry_id"]] = node.available
        if not desired:
            return
        async with self._runtime._context(self.scope_id) as context:
            service = context.artifacts.memory
            artifact_id = context.artifacts.memory_artifact_id
            for wanted_artifact, entries in desired.items():
                if wanted_artifact != artifact_id:
                    raise SourceLifecycleError("memory-identity")
                head = await _head_or_none(service, artifact_id)
                if head is None:
                    continue
                current = {item.entry_id: item.state for item in head.content.manifest.entries}
                versions = {version.entry_id: version for version in await service.entries(head)}
                deactivate = tuple(
                    versions[entry_id]
                    for entry_id, available in entries.items()
                    if not available and current.get(entry_id) == "active" and entry_id in versions
                )
                activate = tuple(
                    versions[entry_id]
                    for entry_id, available in entries.items()
                    if available and current.get(entry_id) == "inactive" and entry_id in versions
                )
                if deactivate:
                    head = await service.forget(head, entries=deactivate, reason="source lifecycle withdrawal")
                if activate:
                    await service.reactivate(head, entries=activate, reason="source lifecycle restore")


async def filter_topic_memory_result(
    runtime: BuiltinRuntime,
    scope_id: str,
    result: Any,
    /,
):
    """Drop Topic Memory hits whose lifecycle availability was withdrawn.

    Applied on the Runtime's scoped Topic Memory search so both the public
    search and Context prepare's topic recall reflect the propagation result.
    Exact ArtifactRef reads stay untouched.
    """

    database = getattr(runtime._provider, "database", None)
    if database is None:
        return result
    async with database.transaction() as connection:
        row = (
            await connection.execute(
                select(SOURCE_LIFECYCLE_TABLE.c.payload).where(SOURCE_LIFECYCLE_TABLE.c.scope_id == scope_id)
            )
        ).first()
    if row is None:
        return result
    state = LifecycleState.model_validate(json.loads(bytes(row[0]).decode("utf-8")))
    inactive = {
        node.ref["artifact_id"]
        for node in state.nodes
        if node.kind == TOPIC_KIND and not node.available
    }
    if not inactive:
        return result
    hits = tuple(hit for hit in result.hits if hit.artifact_ref.artifact_id not in inactive)
    return result.model_copy(update={"hits": hits})


def _database(runtime: BuiltinRuntime):
    database = getattr(runtime._provider, "database", None)
    if database is None:
        raise SourceLifecycleError("source-lifecycle-store")
    return database


async def _head_or_none(service: MemoryService, artifact_id: str) -> Memory | None:
    from powercontext.errors import ArtifactNotFoundError

    try:
        return await service.head(artifact_id)
    except ArtifactNotFoundError:
        return None


def _clean_node_id(node_id: str) -> str:
    if not isinstance(node_id, str) or not node_id or not node_id.strip() or len(node_id) > MAX_NODE_ID_LENGTH:
        raise SourceLifecycleError("node-id")
    return node_id


def _clean_parents(parents: tuple[str, ...]) -> tuple[str, ...]:
    if isinstance(parents, str) or not isinstance(parents, (tuple, list)):
        raise SourceLifecycleError("parents")
    cleaned = tuple(_clean_node_id(parent) for parent in parents)
    if not cleaned:
        raise SourceLifecycleError("parents-empty")
    if len(set(cleaned)) != len(cleaned):
        raise SourceLifecycleError("parents-duplicate")
    return cleaned


def _require_policy(policy: str) -> None:
    if policy not in SUPPORT_POLICIES:
        raise SourceLifecycleError("policy")


def _require_same_binding(
    node: LifecycleNode,
    *,
    kind: str,
    ref: dict[str, Any],
    parents: tuple[str, ...],
    policy: str | None,
) -> None:
    if (
        node.kind != kind
        or node.ref != ref
        or node.parents != parents
        or node.policy != policy
    ):
        raise SourceLifecycleError("node-conflict")


def _require_registered_parents(state: LifecycleState, node_id: str, parents: tuple[str, ...]) -> None:
    if node_id in parents:
        raise SourceLifecycleError("cycle")
    for parent in parents:
        if state.node(parent) is None:
            raise SourceLifecycleError("missing-parent")


def _supported(parents: tuple[str, ...], policy: str, state: LifecycleState) -> bool:
    support = []
    for parent in parents:
        node = state.node(parent)
        if node is None:
            raise SourceLifecycleError("missing-parent")
        support.append(node.available)
    return all(support) if policy == "all" else any(support)


def _propagate(state: LifecycleState) -> None:
    changed = True
    while changed:
        changed = False
        for node in state.nodes:
            if node.kind == SOURCE_NODE_KIND:
                continue
            desired = _supported(node.parents, node.policy or "all", state)
            if node.available != desired:
                node.available = desired
                changed = True


def _citation_ref(citation: MemoryCitation) -> dict[str, Any]:
    return {
        "memory_ref": _artifact_ref_payload(citation.memory_ref),
        "entry_id": citation.entry_id,
        "entry_version_id": citation.entry_version_id,
    }


def _artifact_ref_payload(reference: ArtifactRef) -> dict[str, Any]:
    return {
        "family": reference.family,
        "artifact_id": reference.artifact_id,
        "revision": reference.revision,
    }


def _reference(node: LifecycleNode) -> SourceRef | MemoryCitation | ArtifactRef:
    if node.kind == SOURCE_NODE_KIND:
        return SourceRef(source_type=node.ref["source_type"], source_id=node.ref["source_id"])
    if node.kind == MEMORY_NODE_KIND:
        return MemoryCitation(
            memory_ref=_artifact_ref_from(node.ref["memory_ref"]),
            entry_id=node.ref["entry_id"],
            entry_version_id=node.ref["entry_version_id"],
        )
    return _artifact_ref_from(node.ref)


def _artifact_ref_from(payload: dict[str, Any]) -> ArtifactRef:
    return ArtifactRef(
        family=payload["family"],
        artifact_id=payload["artifact_id"],
        revision=payload["revision"],
    )
