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

"""Pi coding-agent session history: branch-aware capture with durable resume."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, JsonValue, model_validator

from powercontext.sources import TEXT_EVIDENCE_PROJECTION_KEY, AdapterSourceDefinition, TextEvidence
from powercontext.sources.connectors import (
    ConnectorRunCompletion,
    ConnectorRunSession,
    ConnectorRunStatus,
)
from powercontext.sources.models import Source, SourceMaterialization

PI_EVENT_SOURCE_NAME = "pi-event"
PI_EVENT_SOURCE_TYPE = "pi-event"
PI_SESSION_CONNECTOR_NAME = "pi-session-history"
PI_SESSION_CONNECTOR_VERSION = "1"
_IDENTITY_PREFIX = "pi-event:"
_MAX_IDENTITY_LENGTH = 256
_SESSION_FORMAT_VERSION = 3
_CHECKPOINT_VERSION = 1

_SESSION_TYPE = "session"
_MESSAGE_TYPE = "message"
_COMPACTION_TYPE = "compaction"
_BRANCH_SUMMARY_TYPE = "branch_summary"
_CUSTOM_MESSAGE_TYPE = "custom_message"
_CUSTOM_MESSAGE_ROLE = "custom"
_TOOL_RESULT_ROLE = "toolResult"
_BASH_EXECUTION_ROLE = "bashExecution"


class PiSessionInput(BaseModel):
    """Full offline bytes of one Pi session JSONL archive plus the chosen leaf."""

    content: bytes
    leaf_id: str | None = None


class PiEventInput(BaseModel):
    """One visible Pi session event on the selected branch, root first."""

    session_id: str
    event_id: str
    parent_id: str | None
    event_type: str
    role: str
    content: str
    record: dict[str, JsonValue]


class PiEventSource(Source):
    """A single Pi session event captured as exact original evidence."""

    source_id: str
    session_id: str
    event_id: str
    parent_id: str | None
    event_type: str
    role: str
    content: str
    record: dict[str, JsonValue]

    @model_validator(mode="after")
    def validate_identity(self) -> PiEventSource:
        identity = pi_event_source_id(self.session_id, self.event_id)
        if self.source_id != identity or self.name != identity:
            raise ValueError("Pi event identity must match its session_id and event_id")  # noqa: TRY003
        return self


def pi_event_source_id(session_id: str, event_id: str) -> str:
    """Build the stable event identity from the session id and event id alone."""

    digest = hashlib.sha256(f"{session_id}\0{event_id}".encode()).hexdigest()
    identity = f"{_IDENTITY_PREFIX}{digest}"
    if len(identity) > _MAX_IDENTITY_LENGTH:
        raise ValueError("Pi event identity exceeds the maximum source id length")  # noqa: TRY003
    return identity


def _parse_json_line(line: str) -> Any:
    try:
        return json.loads(line)
    except ValueError as error:
        raise ValueError(f"Pi session contains invalid JSON: {error}") from error  # noqa: TRY003


def _render_block(block: Any) -> str | None:
    """Render one message content block into visible text."""

    if not isinstance(block, dict):
        return None
    block_type = block.get("type")
    if block_type == "text":
        text = block.get("text")
        if isinstance(text, str) and text:
            return text
        return None
    if block_type == "toolCall":
        name = block.get("name")
        call_id = block.get("id")
        if not isinstance(name, str) or not name or not isinstance(call_id, str) or not call_id:
            return None
        arguments = block.get("arguments")
        if arguments is None:
            arguments = {}
        rendered = json.dumps(arguments, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return f"[tool_call {name} {call_id}] {rendered}"
    return None  # thinking, image, and other non-visible blocks produce no text.


def _render_content(value: Any) -> str | None:
    """Render a message content string or block list into visible text."""

    if isinstance(value, str):
        return value if value else None
    if not isinstance(value, list):
        return None
    parts = [text for block in value if (text := _render_block(block)) is not None]
    if not parts:
        return None
    return "\n".join(parts)


def _render_tool_result(message: dict[str, Any]) -> tuple[str, str] | None:
    """Render a toolResult message as its tool name, call id, and result text."""

    tool_name = message.get("toolName")
    tool_call_id = message.get("toolCallId")
    if not isinstance(tool_name, str) or not tool_name:
        return None
    if not isinstance(tool_call_id, str) or not tool_call_id:
        return None
    text = _render_content(message.get("content"))
    if text is None:
        return None
    return _TOOL_RESULT_ROLE, f"[tool_result {tool_name} {tool_call_id}] {text}"


def _render_message(message: Any) -> tuple[str, str] | None:
    """Render one message record body, or None without a visible body."""

    if not isinstance(message, dict):
        return None
    role = message.get("role")
    if not isinstance(role, str) or not role:
        return None
    if role == _BASH_EXECUTION_ROLE:
        command = message.get("command")
        if not isinstance(command, str) or not command:
            return None
        body = f"$ {command}"
        output = message.get("output")
        if isinstance(output, str) and output:
            body = f"{body}\n{output}"
        return role, body
    if role == _TOOL_RESULT_ROLE:
        return _render_tool_result(message)
    body = _render_content(message.get("content"))
    if body is None:
        return None
    return role, body


def _render_record(record: dict[str, Any]) -> tuple[str, str] | None:
    """Return the (role, body) pair for one record, or None without visible body."""

    event_type = record.get("type")
    if event_type == _MESSAGE_TYPE:
        return _render_message(record.get("message"))
    if event_type in (_COMPACTION_TYPE, _BRANCH_SUMMARY_TYPE):
        summary = record.get("summary")
        if not isinstance(summary, str) or not summary:
            return None
        return event_type, summary
    if event_type == _CUSTOM_MESSAGE_TYPE:
        body = _render_content(record.get("content"))
        if body is None:
            return None
        return _CUSTOM_MESSAGE_ROLE, body
    return None  # Control events stay on the graph without text evidence.


def _parse_session_header(header: Any) -> str:
    """Validate the mandatory first session record and return its id."""

    if not isinstance(header, dict):
        raise ValueError("Pi session must start with a session header object")  # noqa: TRY003, TRY004
    if header.get("type") != _SESSION_TYPE:
        raise ValueError("Pi session header must declare type=session")  # noqa: TRY003
    version = header.get("version")
    if type(version) is not int or version != _SESSION_FORMAT_VERSION:
        raise ValueError(f"Pi session header must declare version={_SESSION_FORMAT_VERSION}")  # noqa: TRY003
    session_id = header.get("id")
    if not isinstance(session_id, str) or not session_id:
        raise ValueError("Pi session header id must be a non-empty string")  # noqa: TRY003
    return session_id


def _insert_record(
    record: Any,
    *,
    seen: set[str],
    records: dict[str, dict[str, Any]],
    parents: dict[str, str | None],
    order: list[str],
) -> None:
    """Validate one session record and append it to the branch graph."""

    if not isinstance(record, dict):
        raise ValueError("Pi session records must be JSON objects")  # noqa: TRY003, TRY004
    event_id = record.get("id")
    if not isinstance(event_id, str) or not event_id:
        raise ValueError("Pi session record id must be a non-empty string")  # noqa: TRY003
    if event_id in seen:
        raise ValueError(f"Pi session record id {event_id!r} is duplicated")  # noqa: TRY003
    parent_id = record.get("parentId")
    if parent_id is None or parent_id == "":
        parents[event_id] = None
    elif isinstance(parent_id, str) and parent_id in seen:
        parents[event_id] = parent_id
    else:
        raise ValueError(f"Pi session record {event_id!r} references an unknown parent {parent_id!r}")  # noqa: TRY003
    seen.add(event_id)
    records[event_id] = record
    order.append(event_id)


def _select_branch(
    leaf_id: str | None,
    *,
    records: dict[str, dict[str, Any]],
    parents: dict[str, str | None],
    order: list[str],
) -> list[str]:
    """Return the root-to-leaf node chain of the selected branch."""

    if leaf_id is None:
        if not order:
            return []
        leaf_id = order[-1]
    elif leaf_id not in records:
        raise ValueError(f"Pi session leaf {leaf_id!r} does not exist")  # noqa: TRY003
    branch: list[str] = []
    cursor: str | None = leaf_id
    while cursor is not None:
        branch.append(cursor)
        cursor = parents[cursor]
    branch.reverse()
    return branch


def parse_pi_session(value: PiSessionInput) -> tuple[PiEventInput, ...]:
    """Parse one Pi session archive into the selected branch, root to leaf.

    The whole archive is validated before any event is returned; any invalid
    encoding, JSON, graph edge, or missing leaf fails atomically with ValueError.
    """

    try:
        text = value.content.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise ValueError("Pi session is not valid UTF-8") from error  # noqa: TRY003

    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        raise ValueError("Pi session archive is empty")  # noqa: TRY003

    session_id = _parse_session_header(_parse_json_line(lines[0]))
    seen: set[str] = {session_id}
    records: dict[str, dict[str, Any]] = {}
    parents: dict[str, str | None] = {}
    order: list[str] = []
    for line in lines[1:]:
        _insert_record(
            _parse_json_line(line),
            seen=seen,
            records=records,
            parents=parents,
            order=order,
        )

    events: list[PiEventInput] = []
    for event_id in _select_branch(value.leaf_id, records=records, parents=parents, order=order):
        record = records[event_id]
        rendered = _render_record(record)
        if rendered is None:
            continue  # Only events without a visible body are ignored.
        role, body = rendered
        events.append(
            PiEventInput(
                session_id=session_id,
                event_id=event_id,
                parent_id=parents[event_id],
                event_type=record.get("type"),
                role=role,
                content=f"{role}: {body}",
                record=record,
            )
        )
    return tuple(events)


class PiEventSourceAdapter:
    """Resolve and read Pi event Sources."""

    input_class = PiEventInput
    name = PI_EVENT_SOURCE_NAME
    source_class = PiEventSource

    async def resolve(self, value: PiEventInput, /) -> PiEventSource:
        identity = pi_event_source_id(value.session_id, value.event_id)
        return PiEventSource(
            name=identity,
            source_id=identity,
            materialization=SourceMaterialization.CAPTURED,
            session_id=value.session_id,
            event_id=value.event_id,
            parent_id=value.parent_id,
            event_type=value.event_type,
            role=value.role,
            content=value.content,
            record=value.record,
        )

    async def read(self, source: PiEventSource, /) -> PiEventInput:
        return PiEventInput(
            session_id=source.session_id,
            event_id=source.event_id,
            parent_id=source.parent_id,
            event_type=source.event_type,
            role=source.role,
            content=source.content,
            record=source.record,
        )


class PiEventTextEvidenceProjection:
    """Expose the exact visible event text as evidence."""

    name = TEXT_EVIDENCE_PROJECTION_KEY.name
    version = TEXT_EVIDENCE_PROJECTION_KEY.version
    source_class = PiEventSource
    output_class: type[BaseModel] = TextEvidence

    def project(self, source: PiEventSource, /) -> TextEvidence:
        return TextEvidence(
            source_type=PI_EVENT_SOURCE_TYPE,
            source_id=source.source_id,
            content=source.content,
            metadata={
                "session_id": source.session_id,
                "event_id": source.event_id,
                "parent_id": source.parent_id,
                "event_type": source.event_type,
                "role": source.role,
            },
        )


PI_EVENT_SOURCE_ADAPTER = PiEventSourceAdapter()
PI_EVENT_SOURCE_DEFINITION = AdapterSourceDefinition(
    PI_EVENT_SOURCE_ADAPTER,
    projections=(PiEventTextEvidenceProjection(),),
)


def _record_digest(record: dict[str, Any]) -> str:
    canonical = json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


class _ImportedRecords:
    """Durable record of imported Pi events keyed by session and event id."""

    def __init__(self, sessions: dict[str, dict[str, str]]) -> None:
        self._sessions = sessions

    @classmethod
    def from_checkpoint(cls, checkpoint: object, /) -> _ImportedRecords:
        if not isinstance(checkpoint, dict):
            return cls({})
        if checkpoint.get("version") != _CHECKPOINT_VERSION:
            return cls({})
        sessions = checkpoint.get("sessions")
        if not isinstance(sessions, dict):
            return cls({})
        restored: dict[str, dict[str, str]] = {}
        for session_id, events in sessions.items():
            if not isinstance(session_id, str) or not isinstance(events, dict):
                continue
            restored[session_id] = {
                event_id: digest
                for event_id, digest in events.items()
                if isinstance(event_id, str) and isinstance(digest, str)
            }
        return cls(restored)

    def digest(self, session_id: str, event_id: str, /) -> str | None:
        return self._sessions.get(session_id, {}).get(event_id)

    def remember(self, session_id: str, event_id: str, digest: str, /) -> None:
        self._sessions.setdefault(session_id, {})[event_id] = digest

    def to_checkpoint(self) -> JsonValue:
        return {
            "version": _CHECKPOINT_VERSION,
            "sessions": {session_id: dict(events) for session_id, events in sorted(self._sessions.items())},
        }


class PiSessionConnector:
    """Incrementally capture the selected Pi session branch with durable progress.

    Every run fully re-parses the archive and submits only events whose identity
    is not recorded in the durable checkpoint. The checkpoint stores a digest of
    each imported record so rewritten history fails without advancing progress.
    """

    name = PI_SESSION_CONNECTOR_NAME
    version = PI_SESSION_CONNECTOR_VERSION
    source_definitions = frozenset({PI_EVENT_SOURCE_NAME})

    def __init__(self, path: str | Path, leaf_id: str | None = None) -> None:
        self._path = Path(path)
        self._leaf_id = leaf_id

    async def run(self, session: ConnectorRunSession, /) -> ConnectorRunCompletion:
        raw = self._path.read_bytes()
        events = parse_pi_session(PiSessionInput(content=raw, leaf_id=self._leaf_id))
        imported = _ImportedRecords.from_checkpoint(session.checkpoint)

        for event in events:  # Validate all imported history before submitting anything new.
            known = imported.digest(event.session_id, event.event_id)
            if known is not None and known != _record_digest(event.record):
                raise ValueError(f"imported Pi event {event.event_id!r} was rewritten")  # noqa: TRY003

        for event in events:
            if imported.digest(event.session_id, event.event_id) is not None:
                continue
            identity = pi_event_source_id(event.session_id, event.event_id)
            await session.submit(identity, PI_EVENT_SOURCE_NAME, event)
            imported.remember(event.session_id, event.event_id, _record_digest(event.record))

        return ConnectorRunCompletion(
            status=ConnectorRunStatus.COMPLETE,
            checkpoint=imported.to_checkpoint(),
        )


__all__ = [
    "PI_EVENT_SOURCE_DEFINITION",
    "PI_EVENT_SOURCE_NAME",
    "PI_EVENT_SOURCE_TYPE",
    "PI_SESSION_CONNECTOR_NAME",
    "PI_SESSION_CONNECTOR_VERSION",
    "PiEventInput",
    "PiEventSource",
    "PiSessionConnector",
    "PiSessionInput",
    "parse_pi_session",
    "pi_event_source_id",
]
