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

"""seekdb Memory search: one native statement for hybrid candidates."""

from __future__ import annotations

from pyobvector import VECTOR
from sqlalchemy import bindparam, text
from sqlalchemy.ext.asyncio import AsyncConnection

from powercontext.builtin.artifacts.memory import (
    CapabilityNotSupportedError,
    MemorySearchChannels,
    MemorySearchRequest,
)
from powercontext.builtin.artifacts.memory.canonical import validate_embedding
from powercontext.builtin.persistence.memory_index import (
    CompositeMemoryIndex,
    MemoryIndex,
    memory_channel_hits,
)
from powercontext.builtin.persistence.oceanbase.memory_index import (
    OceanBaseMemoryFTSIndex,
    OceanBaseMemoryVectorIndex,
)
from powercontext.builtin.persistence.tags import memory_tag_parameters, memory_tag_sql

_HYBRID_FTS_CHANNEL = 0
_HYBRID_VECTOR_CHANNEL = 1

# One candidate statement per hybrid search: the full-text branch and the
# vector branch each keep their native index order and candidate limit, and
# the channel marker lets the fusion layer reassemble both rankings from the
# single round-trip without re-querying.
_HYBRID_FTS_BRANCH = """
(SELECT
    h.memory_artifact_id,
    h.head_revision,
    h.entry_id,
    h.entry_version_id,
    v.text,
    NULL AS distance,
    0 AS channel
 FROM pc_memory_entry_heads AS h
 JOIN pc_memory_entry_versions AS v
   ON v.scope_id = h.scope_id
  AND v.memory_artifact_id = h.memory_artifact_id
  AND v.entry_version_id = h.entry_version_id
 WHERE h.scope_id = :scope_id
   AND h.memory_artifact_id IN :memory_artifact_ids
   AND MATCH(h.searchable_text) AGAINST (:fts_query IN NATURAL LANGUAGE MODE)
   {tag_predicate}
 ORDER BY MATCH(h.searchable_text) AGAINST (:fts_query IN NATURAL LANGUAGE MODE) DESC,
          h.memory_artifact_id,
          h.entry_id,
          h.entry_version_id
 LIMIT :candidate_limit)
"""

_HYBRID_VECTOR_BRANCH = """
(SELECT
    m.memory_artifact_id,
    m.head_revision,
    m.entry_id,
    m.entry_version_id,
    v.text,
    l2_distance(m.embedding, :query_vector) AS distance,
    1 AS channel
 FROM pc_memory_vector_entries AS m
 JOIN pc_memory_entry_versions AS v
   ON v.scope_id = m.scope_id
  AND v.memory_artifact_id = m.memory_artifact_id
  AND v.entry_version_id = m.entry_version_id
 WHERE m.scope_id = :scope_id
   AND m.memory_artifact_id IN :memory_artifact_ids
   {tag_predicate}
 ORDER BY l2_distance(m.embedding, :query_vector){approximate}
 LIMIT :candidate_limit)
"""


class SeekDBMemoryIndex(CompositeMemoryIndex):
    """Composite Memory index whose hybrid mode runs one native seekdb statement.

    The full-text and vector single-channel searches keep the shared composite
    behaviour bit for bit; only the hybrid candidate retrieval is fused into a
    single statement that uses both native indexes in the database.
    """

    def __init__(self, *indexes: MemoryIndex) -> None:
        super().__init__(*indexes)
        self._fts_index = next(index for index in indexes if index.capabilities.fts)
        self._vector_index = next(index for index in indexes if index.capabilities.vector)
        if not isinstance(self._fts_index, OceanBaseMemoryFTSIndex) or not isinstance(
            self._vector_index, OceanBaseMemoryVectorIndex
        ):
            raise CapabilityNotSupportedError("hybrid", "seekdb hybrid requires the native full-text and vector indexes")
        self._hybrid_sql = text(
            _HYBRID_FTS_BRANCH.format(tag_predicate="", approximate=" APPROXIMATE")
            + "UNION ALL"
            + _HYBRID_VECTOR_BRANCH.format(tag_predicate="", approximate=" APPROXIMATE")
        ).bindparams(
            bindparam("memory_artifact_ids", expanding=True),
            bindparam("query_vector", type_=VECTOR(self._vector_index.profile.dimension)),
        )
        self._tagged_hybrid_sql = text(
            _HYBRID_FTS_BRANCH.format(tag_predicate=memory_tag_sql("h"), approximate="")
            + "UNION ALL"
            + _HYBRID_VECTOR_BRANCH.format(tag_predicate=memory_tag_sql("m"), approximate="")
        ).bindparams(
            bindparam("memory_artifact_ids", expanding=True),
            bindparam("tag_keys", expanding=True),
            bindparam("tag_hashes", expanding=True),
            bindparam("query_vector", type_=VECTOR(self._vector_index.profile.dimension)),
        )

    async def search(
        self,
        connection: AsyncConnection,
        scope_id: str,
        request: MemorySearchRequest,
        /,
    ) -> MemorySearchChannels:
        if request.mode != "hybrid":
            return await super().search(connection, scope_id, request)
        if request.embedding_profile != self._vector_index.profile or request.query_vector is None:
            raise CapabilityNotSupportedError("embedding-profile")
        if not await self._vector_index.vector_complete(
            connection, scope_id, request.memories, self._vector_index.profile
        ):
            raise CapabilityNotSupportedError("vector")
        rows = (
            await connection.execute(
                self._hybrid_sql if request.tag_filter is None else self._tagged_hybrid_sql,
                {
                    "scope_id": scope_id,
                    "memory_artifact_ids": tuple(memory.artifact_id for memory in request.memories),
                    "fts_query": request.analyzed_query,
                    "query_vector": validate_embedding(
                        request.query_vector,
                        dimension=self._vector_index.profile.dimension,
                    ),
                    "candidate_limit": request.candidate_limit,
                    **memory_tag_parameters(request.tag_filter),
                },
            )
        ).mappings().all()
        # The statement returns the full-text ranking first, then the vector
        # ranking; splitting on the channel marker preserves each branch order.
        return MemorySearchChannels(
            fts=memory_channel_hits(
                tuple(row for row in rows if row["channel"] == _HYBRID_FTS_CHANNEL), request.memories
            ),
            vector=memory_channel_hits(
                tuple(row for row in rows if row["channel"] == _HYBRID_VECTOR_CHANNEL), request.memories
            ),
        )
