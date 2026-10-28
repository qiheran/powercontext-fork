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

"""Row-level CSV capture with exact original-record evidence."""

from __future__ import annotations

import csv
import io
import json
from typing import TYPE_CHECKING
from urllib.parse import quote

from pydantic import BaseModel, model_validator

from powercontext.limits import MAX_SOURCE_ID_LENGTH
from powercontext.sources import TEXT_EVIDENCE_PROJECTION_KEY, AdapterSourceDefinition, TextEvidence
from powercontext.sources.models import Source, SourceMaterialization

if TYPE_CHECKING:
    from collections.abc import Iterator

CSV_ROW_SOURCE_NAME = "csv-row"
CSV_ROW_SOURCE_TYPE = "csv-row"
_IDENTITY_PREFIX = "csv:"


class CSVDocumentInput(BaseModel):
    """Full offline bytes of one CSV document plus its identity and primary key."""

    document_id: str
    primary_key: str
    content: bytes


class CSVRowInput(BaseModel):
    """One captured CSV row preserving the original header order and field values."""

    document_id: str
    primary_key: str
    primary_key_value: str
    columns: tuple[str, ...]
    values: tuple[str, ...]


class CSVRowSource(Source):
    """A single CSV row captured as exact original evidence."""

    source_id: str
    document_id: str
    primary_key: str
    primary_key_value: str
    columns: tuple[str, ...]
    values: tuple[str, ...]

    @model_validator(mode="after")
    def validate_identity(self) -> CSVRowSource:
        identity = csv_row_source_id(self.document_id, self.primary_key_value)
        if self.source_id != identity or self.name != identity:
            raise ValueError("CSV row identity must match its document_id and primary_key_value")  # noqa: TRY003
        return self


def csv_row_source_id(document_id: str, primary_key_value: str) -> str:
    """Build the stable row identity, percent-encoding both components."""
    if not isinstance(document_id, str) or not document_id or document_id != document_id.strip():
        raise ValueError("document_id must be non-empty without leading or trailing whitespace")  # noqa: TRY003
    identity = (
        f"{_IDENTITY_PREFIX}{quote(document_id, safe='')}:"
        f"{quote(primary_key_value, safe='')}"
    )
    if len(identity) > MAX_SOURCE_ID_LENGTH:
        raise ValueError("CSV row identity exceeds the maximum source id length")  # noqa: TRY003
    return identity


def _iter_records(text: str) -> Iterator[list[str]]:
    reader = csv.reader(io.StringIO(text, newline=""), strict=True)
    try:
        yield from reader
    except csv.Error as error:
        raise ValueError(f"malformed CSV document: {error}") from error


def parse_csv_document(value: CSVDocumentInput) -> tuple[CSVRowInput, ...]:
    """Parse a whole CSV document into rows, failing atomically on any error."""
    document_id = value.document_id
    if not document_id or document_id != document_id.strip():
        raise ValueError("document_id must be non-empty without leading or trailing whitespace")  # noqa: TRY003
    primary_key = value.primary_key
    if not primary_key:
        raise ValueError("primary_key must be non-empty")  # noqa: TRY003

    try:
        text = value.content.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise ValueError("CSV document is not valid UTF-8") from error

    records = list(_iter_records(text))
    if not records:
        raise ValueError("CSV document is missing its header record")
    header = records[0]
    if not header:
        raise ValueError("CSV header record is empty")  # noqa: TRY003
    for column in header:
        if not column or not column.strip():
            raise ValueError("CSV header fields must be non-empty and not blank")  # noqa: TRY003
    if len(set(header)) != len(header):
        raise ValueError("CSV header fields must be unique")  # noqa: TRY003
    if primary_key not in header:
        raise ValueError("primary_key must name one of the CSV header columns")  # noqa: TRY003
    key_index = header.index(primary_key)

    rows: list[CSVRowInput] = []
    seen_keys: set[str] = set()
    columns = tuple(header)
    for record in records[1:]:
        if not record:
            continue  # Fully blank physical records after the header are ignored.
        if len(record) != len(columns):
            raise ValueError("CSV data record width does not match the header")  # noqa: TRY003
        primary_key_value = record[key_index]
        if not primary_key_value.strip():
            raise ValueError("CSV primary key value must not be empty or blank")  # noqa: TRY003
        if primary_key_value in seen_keys:
            raise ValueError("CSV primary key values must be unique within the document")  # noqa: TRY003
        seen_keys.add(primary_key_value)
        csv_row_source_id(document_id, primary_key_value)  # Identity must stay bounded.
        rows.append(
            CSVRowInput(
                document_id=document_id,
                primary_key=primary_key,
                primary_key_value=primary_key_value,
                columns=columns,
                values=tuple(record),
            )
        )
    return tuple(rows)


class CSVRowSourceAdapter:
    """Resolve and read row-level CSV Sources."""

    input_class = CSVRowInput
    name = CSV_ROW_SOURCE_NAME
    source_class = CSVRowSource

    async def resolve(self, value: CSVRowInput, /) -> CSVRowSource:
        source_id = csv_row_source_id(value.document_id, value.primary_key_value)
        return CSVRowSource(
            name=source_id,
            source_id=source_id,
            materialization=SourceMaterialization.CAPTURED,
            document_id=value.document_id,
            primary_key=value.primary_key,
            primary_key_value=value.primary_key_value,
            columns=value.columns,
            values=value.values,
        )

    async def read(self, source: CSVRowSource, /) -> CSVRowInput:
        return CSVRowInput(
            document_id=source.document_id,
            primary_key=source.primary_key,
            primary_key_value=source.primary_key_value,
            columns=source.columns,
            values=source.values,
        )


class CSVRowTextEvidenceProjection:
    """Expose the exact original record as textual evidence."""

    name = TEXT_EVIDENCE_PROJECTION_KEY.name
    version = TEXT_EVIDENCE_PROJECTION_KEY.version
    source_class = CSVRowSource
    output_class: type[BaseModel] = TextEvidence

    def project(self, source: CSVRowSource, /) -> TextEvidence:
        row = dict(zip(source.columns, source.values, strict=True))
        return TextEvidence(
            source_type=CSV_ROW_SOURCE_TYPE,
            source_id=source.source_id,
            content=json.dumps(row, ensure_ascii=False, separators=(",", ":")),
            metadata={
                "document_id": source.document_id,
                "primary_key": source.primary_key,
                "primary_key_value": source.primary_key_value,
                "columns": list(source.columns),
                "row": row,
            },
        )


CSV_ROW_SOURCE_ADAPTER = CSVRowSourceAdapter()
CSV_ROW_SOURCE_DEFINITION = AdapterSourceDefinition(
    CSV_ROW_SOURCE_ADAPTER,
    projections=(CSVRowTextEvidenceProjection(),),
)
