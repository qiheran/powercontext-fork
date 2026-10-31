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

"""Whole-document PDF page capture with page-level evidence retrieval."""

from __future__ import annotations

import hashlib
import io
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import quote

from pydantic import BaseModel, model_validator

from powercontext.limits import MAX_SOURCE_ID_LENGTH
from powercontext.sources import TEXT_EVIDENCE_PROJECTION_KEY, AdapterSourceDefinition, TextEvidence
from powercontext.sources.models import Source, SourceMaterialization

PDF_PAGE_SOURCE_NAME = "pdf-page"
PDF_PAGE_SOURCE_TYPE = "pdf-page"
_IDENTITY_PREFIX = "pdf:"
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


class PDFDocumentInput(BaseModel):
    """Full offline bytes of one PDF document plus its identity."""

    document_id: str
    content: bytes


class PDFPageInput(BaseModel):
    """One captured physical PDF page preserving the whole-file digest identity."""

    document_id: str
    document_sha256: str
    page_number: int
    text: str

    @model_validator(mode="after")
    def validate_page(self) -> PDFPageInput:
        _require_document_id(self.document_id)
        if not isinstance(self.document_sha256, str) or not _SHA256_PATTERN.fullmatch(self.document_sha256):
            raise ValueError("document_sha256 must be a lowercase SHA-256 hex digest")  # noqa: TRY003
        if self.page_number < 1:
            raise ValueError("page_number must be a one-based physical page number")  # noqa: TRY003
        if not isinstance(self.text, str):
            raise ValueError("text must be a string")  # noqa: TRY003
        return self


def _require_document_id(document_id: object) -> None:
    if not isinstance(document_id, str) or not document_id.strip():
        raise ValueError("document_id must be non-empty and not pure whitespace")  # noqa: TRY003


def pdf_page_source_id(document_id: str, document_sha256: str, page_number: int) -> str:
    """Build the stable page identity from the document digest and physical page."""
    _require_document_id(document_id)
    if not isinstance(document_sha256, str) or not _SHA256_PATTERN.fullmatch(document_sha256):
        raise ValueError("document_sha256 must be a lowercase SHA-256 hex digest")  # noqa: TRY003
    if not isinstance(page_number, int) or page_number < 1:
        raise ValueError("page_number must be a one-based physical page number")  # noqa: TRY003
    identity = f"{_IDENTITY_PREFIX}{quote(document_id, safe='')}:{document_sha256}:{page_number}"
    if len(identity) > MAX_SOURCE_ID_LENGTH:
        raise ValueError("PDF page identity exceeds the maximum source id length")  # noqa: TRY003
    return identity


def parse_pdf_document(value: PDFDocumentInput) -> tuple[PDFPageInput, ...]:
    """Parse a whole PDF into every physical page, failing atomically on any error."""
    _require_document_id(value.document_id)
    try:
        pages = tuple(_extract_pages(value.document_id, value.content))
    except Exception as error:
        raise ValueError(f"PDF document {value.document_id!r} cannot be parsed: {error}") from error
    if not pages:
        raise ValueError(f"PDF document {value.document_id!r} contains no pages")  # noqa: TRY003
    return pages


def _extract_pages(document_id: str, content: bytes) -> Any:
    # pypdf is provided by the locked shared environment; import lazily so the module
    # stays importable in installs that never touch PDF capture.
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(content))
    if reader.is_encrypted:
        if not reader.decrypt(""):
            raise ValueError("PDF document requires a password")  # noqa: TRY003
    document_sha256 = hashlib.sha256(content).hexdigest()
    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if not text.strip():
            text = ""
        yield PDFPageInput(
            document_id=document_id,
            document_sha256=document_sha256,
            page_number=page_number,
            text=text,
        )


class PDFPageSource(Source):
    """A single physical PDF page captured as exact page-level evidence."""

    document_id: str
    document_sha256: str
    page_number: int
    text: str

    @model_validator(mode="after")
    def validate_identity(self) -> PDFPageSource:
        identity = pdf_page_source_id(self.document_id, self.document_sha256, self.page_number)
        if self.name != identity:
            raise ValueError("PDF page identity must match its document digest and physical page")  # noqa: TRY003
        return self


class PDFPageSourceAdapter:
    """Resolve and read page-level PDF Sources."""

    input_class = PDFPageInput
    name = PDF_PAGE_SOURCE_NAME
    source_class = PDFPageSource

    async def resolve(self, value: PDFPageInput, /) -> PDFPageSource:
        source_id = pdf_page_source_id(value.document_id, value.document_sha256, value.page_number)
        return PDFPageSource(
            name=source_id,
            materialization=SourceMaterialization.CAPTURED,
            document_id=value.document_id,
            document_sha256=value.document_sha256,
            page_number=value.page_number,
            text=value.text,
        )

    async def read(self, source: PDFPageSource, /) -> PDFPageInput:
        return PDFPageInput(
            document_id=source.document_id,
            document_sha256=source.document_sha256,
            page_number=source.page_number,
            text=source.text,
        )


class PDFPageTextEvidenceProjection:
    """Expose the full page text with its document digest and physical page."""

    name = TEXT_EVIDENCE_PROJECTION_KEY.name
    version = TEXT_EVIDENCE_PROJECTION_KEY.version
    source_class = PDFPageSource
    output_class: type[BaseModel] = TextEvidence

    def project(self, source: PDFPageSource, /) -> TextEvidence:
        return TextEvidence(
            source_type=PDF_PAGE_SOURCE_TYPE,
            source_id=source.name,
            content=f"{source.document_id}\n{source.text}",
            metadata={
                "document_id": source.document_id,
                "document_sha256": source.document_sha256,
                "page_number": source.page_number,
            },
        )


PDF_PAGE_SOURCE_ADAPTER = PDFPageSourceAdapter()
PDF_PAGE_SOURCE_DEFINITION = AdapterSourceDefinition(
    PDF_PAGE_SOURCE_ADAPTER,
    projections=(PDFPageTextEvidenceProjection(),),
)


# ---------------------------------------------------------------------------
# Offline evidence-page retrieval over one scoped Memory.
#
# Every candidate page comes from the public Memory API; scoring below only
# re-ranks the collected entries with a deterministic BM25-style lexical model.
# ---------------------------------------------------------------------------

_WORD_PATTERN = re.compile(r"[a-z]+|\d+")

_FISCAL_YEAR_PATTERN = re.compile(r"\bf(?:y|iscal)\s*(\d{2,4})\b")
_AMORTIZATION_ACRONYM_PATTERN = re.compile(r"\bd\s*&\s*a\b")
_MA_ACRONYM_PATTERN = re.compile(r"\bm\s*&\s*a\b")
_INCOME_ACRONYM_PATTERN = re.compile(r"\bp\s*&\s*l\b")


def _strip_plural(token: str) -> str:
    if token.endswith("ies") and len(token) > 4:
        return f"{token[:-3]}y"
    if token.endswith("sses"):
        return token[:-2]
    if token.endswith("s") and not token.endswith(("ss", "us", "is")):
        return token[:-1]
    return token

def _content_stems(tokens: list[str]) -> list[str]:
    return [_stem(token) for token in tokens if token not in _STOPWORDS and (len(token) > 1 or token.isdigit())]


def _bigram_key(first: str, second: str) -> tuple[str, str]:
    return (first, second) if first <= second else (second, first)


_HEADER_BOILERPLATE = frozenset(("table", "contents", "content", "page", "able", "10", "k"))



_EXPANSION_WEIGHT = 0.5
_BIGRAM_BONUS = 2.0
_HEADER_WINDOW = 40
_HEADER_BONUS = 12.0
_HEADER_CONCEPT_BONUS = 14.0
_DOCUMENT_IDENTITY_BONUS = 14.0
_STATEMENT_PAGE_BONUS = 12.0
_COVERAGE_BONUS = 6.0
_BM25_K1 = 1.5
_BM25_B = 0.35

_STOPWORDS = frozenset((
    "a", "an", "the", "and", "or", "of", "to", "in", "on", "at", "by", "for", "from", "with",
    "as", "is", "are", "was", "were", "be", "been", "being", "what", "which", "who", "how",
    "why", "when", "where", "do", "does", "did", "can", "could", "should", "would", "will",
    "shall", "may", "might", "must", "we", "you", "it", "its", "our", "your", "this", "that",
    "these", "those", "their", "they", "them", "his", "her", "not", "no", "yes", "if", "then",
    "than", "so", "such", "any", "all", "each", "every", "both", "has", "have", "had", "having",
    "also", "into", "only", "per", "via", "please", "about", "over", "under", "between",
    "within", "without", "whether", "there", "here", "other", "more", "most", "less", "least",
    "very", "just", "like", "same", "own",
    "reasonably", "healthy", "improving", "improve", "improved", "dragged",
    "subject", "based", "data", "compare", "comparing", "report", "reports",
    "excluding", "exclusion", "include", "including", "given", "well",
))

_FISCAL_YEAR_TOKEN = re.compile(r"^fy(\d{2,4})$")


def _full_year(digits: str) -> str:
    return digits if len(digits) == 4 else f"20{digits}"

def _strip_suffixes(token: str) -> str:
    if token.endswith("ation") and len(token) > 6:
        token = token[:-2]
    if token.endswith("ing") and len(token) > 5:
        return token[:-3]
    if token.endswith("ed") and len(token) > 4:
        token = token[:-2]
    if token.endswith("ity") and len(token) > 6:
        token = token[:-3]
    if token.endswith("e") and len(token) > 4:
        return token[:-1]
    return token


def _stem(token: str) -> str:
    if len(token) <= 3 or token.isdigit():
        return token
    return _strip_suffixes(_strip_plural(token))





def _hdr(first: str, second: str) -> tuple[str, str]:
    return _bigram_key(_stem(first), _stem(second))


_PRIMARY_STATEMENT_BIGRAMS = frozenset((
    _hdr("consolidated", "statements"),
    _hdr("statement", "operations"),
    _hdr("statement", "income"),
    _hdr("balance", "sheets"),
    _hdr("statements", "cash"),
    _hdr("cash", "flows"),
))


_TOKEN_EXPANSIONS: dict[str, tuple[str, ...]] = {
    "fy": ("fiscal",),
    "ebitda": ("operating", "income", "depreciation", "amortization"),
    "capex": ("capital", "spending", "expenditures"),
    "geographies": (
        "geography", "divisions", "segments", "regions", "businesses",
        "north", "america", "europe", "latin", "international",
    ),
    "geography": ("divisions", "segments", "regions", "businesses"),
    "battles": (
        "litigation", "proceedings", "contingencies", "disputes", "lawsuits",
        "inquiries", "investigations",
    ),
    "battle": (
        "litigation", "proceedings", "contingencies", "disputes", "lawsuits",
        "inquiries", "investigations",
    ),
    "lawsuits": ("litigation", "proceedings", "contingencies", "disputes", "inquiries", "investigations"),
    "lawsuit": ("litigation", "proceedings", "contingencies", "disputes", "inquiries", "investigations"),
    "litigation": ("proceedings", "contingencies", "disputes", "inquiries", "investigations"),
    "restructuring": ("charges", "impairment"),
    "revenues": ("sales",),
    "revenue": ("sales",),
    "quantity": ("amount", "total"),
    "operating": ("operations",),
    "depreciation": ("amortization",),
    "amortization": ("depreciation",),
    "statement": ("statements",),
}

_PHRASE_EXPANSIONS: tuple[tuple[frozenset[str], tuple[str, ...]], ...] = (
    (frozenset(("income", "statement")), ("consolidated",)),
    (frozenset(_hdr("cash", "flows")), _hdr("consolidated", "statements")),
    (frozenset(_hdr("balance", "sheets")), _hdr("consolidated", "statements")),
    (frozenset(("operating", "income")), ("profit", "operations")),
    (frozenset(("capital", "intensive")), (
        "depreciation", "property", "plant", "equipment", "expenditures", "purchases",
    )),
    (frozenset(("days", "payable")), ("accounts", "payable", "inventory", "purchases")),
    (frozenset(("customer", "concentration")), ("customers", "revenue", "significant")),
    (frozenset(("effective", "tax")), ("income", "taxes", "rate")),
    (frozenset(("gross", "margin")), ("cost", "sales", "profit")),
    (frozenset(("working", "capital")), ("current", "assets", "liabilities", "cash")),
)

# Concept groups map finance questions to the vocabulary and statement headers
# that typically carry the answer, since 10-K evidence concentrates on the
# primary financial statements.
_CONCEPT_GROUPS: tuple[tuple[frozenset[str], tuple[str, ...], tuple[tuple[str, str], ...]], ...] = (
    (
        frozenset(("liquidity", "liquid", "solvency", "balance")),
        ("cash", "current", "assets", "liabilities", "investments", "debt"),
        (_hdr("balance", "sheets"), _hdr("consolidated", "statements"), _hdr("statements", "cash")),
    ),
    (
        frozenset(("capex", "capital", "expenditure", "expenditures")),
        ("depreciation", "amortization", "property", "plant", "equipment", "purchases", "spending"),
        (_hdr("cash", "flows"), _hdr("statements", "cash"), _hdr("consolidated", "statements"),
         _hdr("statement", "income"), _hdr("statement", "operations"), _hdr("balance", "sheets")),
    ),
    (
        frozenset(("revenue", "revenues", "sales")),
        ("net", "sales", "products", "services", "cost"),
        (_hdr("statement", "operations"), _hdr("consolidated", "statements"), _hdr("statement", "income")),
    ),
    (
        frozenset(("cyclicality", "cyclical", "cycle")),
        ("cyclical", "demand", "downturn", "defense", "commercial", "economic"),
        (),
    ),
    (
        frozenset(("customers", "customer", "concentration")),
        ("customers", "concentration", "significant", "revenue", "accounted"),
        (),
    ),
    (
        frozenset(("dpo", "payable", "payables")),
        ("accounts", "payable", "inventory", "purchases", "cost", "sales"),
        (_hdr("balance", "sheets"), _hdr("statement", "operations"), _hdr("consolidated", "statements")),
    ),
    (
        frozenset(("tax", "taxes", "taxpayer")),
        ("income", "taxes", "effective", "rate", "provision"),
        (_hdr("income", "taxes"), _hdr("statement", "income"), _hdr("statement", "operations")),
    ),
    (
        frozenset(("margin", "margins")),
        ("profit", "operating", "gross", "cost", "sales"),
        (_hdr("statement", "operations"), _hdr("statement", "income"), _hdr("consolidated", "statements")),
    ),
    (
        frozenset(("segment", "segments")),
        ("segment", "operating", "reportable", "business"),
        (),
    ),
    (
        frozenset(("inventory", "inventories")),
        ("inventories", "balance", "current", "assets"),
        (_hdr("balance", "sheets"),),
    ),
    (
        frozenset(("debt", "borrowings")),
        ("debt", "borrowings", "loans", "notes", "long", "term"),
        (_hdr("balance", "sheets"),),
    ),
    (
        frozenset(("repair", "repairs", "maintenance", "aftermarket")),
        ("products", "services", "spare", "parts", "aftermarket", "replacement"),
        (),
    ),
    (
        frozenset(("gross", "profit")),
        ("cost", "sales", "margin", "profit"),
        (_hdr("statement", "operations"), _hdr("statement", "income")),
    ),
    (
        frozenset(("ebitda",)),
        ("operating", "income", "depreciation", "amortization", "profit"),
        (_hdr("statement", "income"), _hdr("statement", "operations"), _hdr("consolidated", "statements")),
    ),
)

_TERM_CACHE_LIMIT = 4096
_TERM_CACHE: dict[str, tuple[Counter[str], Counter[tuple[str, str]], Counter[tuple[str, str]], int, bool]] = {}


def _document_statistics(
    text: str,
) -> tuple[Counter[str], Counter[tuple[str, str]], Counter[tuple[str, str]], int, bool]:
    """Tokenize one entry text into BM25 term, body-bigram and header-bigram counts."""
    cached = _TERM_CACHE.get(text)
    if cached is not None:
        return cached
    stems = _content_stems(_WORD_PATTERN.findall(text.lower()))
    terms: Counter[str] = Counter(stems)
    bigrams: Counter[tuple[str, str]] = Counter(
        _bigram_key(first, second) for first, second in zip(stems, stems[1:])
    )
    lead = [stem for stem in stems if stem not in _HEADER_BOILERPLATE][:_HEADER_WINDOW]
    lead = [_strip_plural(stem) for stem in lead]
    header: Counter[tuple[str, str]] = Counter(
        _bigram_key(first, second) for first, second in zip(lead, lead[1:])
    )
    is_primary_statement = any(bigram in _PRIMARY_STATEMENT_BIGRAMS for bigram in header)
    statistics = (terms, bigrams, header, len(stems), is_primary_statement)
    if len(_TERM_CACHE) >= _TERM_CACHE_LIMIT:
        _TERM_CACHE.clear()
    _TERM_CACHE[text] = statistics
    return statistics


@dataclass(frozen=True)
class _QueryPlan:
    terms: tuple[tuple[str, float], ...]
    bigrams: tuple[tuple[str, str], ...]
    concept_header_bigrams: tuple[tuple[str, str], ...] = ()


def _build_query_plan(question: str) -> _QueryPlan:
    tokens = _WORD_PATTERN.findall(question.lower())
    counts: Counter[str] = Counter(_content_stems(tokens))
    lowered = question.lower()
    for match in _FISCAL_YEAR_PATTERN.finditer(lowered):
        counts[_full_year(match.group(1))] += _EXPANSION_WEIGHT
    for token in tokens:
        fiscal = _FISCAL_YEAR_TOKEN.fullmatch(token)
        if fiscal is not None:
            counts[_full_year(fiscal.group(1))] += _EXPANSION_WEIGHT
            counts["fiscal"] += _EXPANSION_WEIGHT
        additions = _TOKEN_EXPANSIONS.get(token)
        if additions:
            for addition in additions:
                counts[_stem(addition)] += _EXPANSION_WEIGHT
    if _AMORTIZATION_ACRONYM_PATTERN.search(lowered):
        counts[_stem("depreciation")] += _EXPANSION_WEIGHT
        counts[_stem("amortization")] += _EXPANSION_WEIGHT
    if _INCOME_ACRONYM_PATTERN.search(lowered):
        counts[_stem("income")] += _EXPANSION_WEIGHT
        counts[_stem("operations")] += _EXPANSION_WEIGHT
    stems = _content_stems(tokens)
    stem_set = set(stems)
    for phrase, additions in _PHRASE_EXPANSIONS:
        if phrase <= stem_set:
            for addition in additions:
                counts[_stem(addition)] += _EXPANSION_WEIGHT
    expanded_bigrams: list[tuple[str, str]] = []
    if _MA_ACRONYM_PATTERN.search(lowered):
        counts[_stem("acquisitions")] += _EXPANSION_WEIGHT
        counts[_stem("divestitures")] += _EXPANSION_WEIGHT
        counts[_stem("portfolio")] += _EXPANSION_WEIGHT
    if _AMORTIZATION_ACRONYM_PATTERN.search(lowered):
        expanded_bigrams.append(_bigram_key(_stem("depreciation"), _stem("amortization")))
    if _INCOME_ACRONYM_PATTERN.search(lowered):
        expanded_bigrams.append(_bigram_key(_stem("income"), _stem("statement")))
    bigrams: list[tuple[str, str]] = [
        _bigram_key(first, second) for first, second in zip(stems, stems[1:])
    ]
    bigrams.extend(expanded_bigram for expanded_bigram in expanded_bigrams if expanded_bigram not in bigrams)
    concept_header_bigrams: list[tuple[str, str]] = []
    stem_set = set(stems)
    for indicators, additions, headers in _CONCEPT_GROUPS:
        if indicators & stem_set:
            for addition in additions:
                counts[_stem(addition)] += _EXPANSION_WEIGHT
            for header in headers:
                if header not in concept_header_bigrams:
                    concept_header_bigrams.append(header)
    return _QueryPlan(
        terms=tuple(counts.items()),
        bigrams=tuple(dict.fromkeys(bigrams)),
        concept_header_bigrams=tuple(concept_header_bigrams),
    )


@dataclass
class _ScoredDocument:
    """One Memory entry version carrying its page text and retrieval statistics."""

    key: tuple[str, str]
    memory_ref: Any
    text: str
    source_ids: tuple[str, ...]
    score: float = 0.0
    terms: Counter[str] = field(default_factory=Counter)
    bigrams: Counter[tuple[str, str]] = field(default_factory=Counter)
    header: Counter[tuple[str, str]] = field(default_factory=Counter)
    length: int = 0
    is_primary_statement: bool = False


async def _collect_documents(memory: Any, question: str, limit: int) -> list[_ScoredDocument]:
    """Collect candidate entries through the public Memory search and listing APIs."""
    from powercontext.builtin.runtime.models import SearchMemoryRequest

    documents: dict[tuple[str, str], _ScoredDocument] = {}
    pool_limit = max(limit * 4, 48)
    try:
        page = await memory.search(SearchMemoryRequest(query=question, mode="fts", limit=pool_limit))
    except Exception:  # noqa: BLE001 - search stays best-effort behind the listing fallback
        page = None
    if page is not None:
        for hit in page.hits:
            key = (hit.entry_id, hit.entry_version_id)
            documents[key] = _ScoredDocument(
                key=key,
                memory_ref=hit.memory_ref,
                text=hit.text,
                source_ids=(),
            )
    try:
        listing = await memory.list()
    except Exception:  # noqa: BLE001 - retrieval still works from the search pool
        listing = None
    if listing is not None and listing.memory_ref is not None:
        for record in listing.entries:
            key = (record.entry.entry_id, record.entry.entry_version_id)
            documents[key] = _ScoredDocument(
                key=key,
                memory_ref=record.memory_ref,
                text=record.entry.text,
                source_ids=tuple(reference.source_id for reference in record.entry.sources),
            )
    collected: list[_ScoredDocument] = []
    for document in documents.values():
        if not document.text.strip():
            continue
        (
            document.terms,
            document.bigrams,
            document.header,
            document.length,
            document.is_primary_statement,
        ) = _document_statistics(document.text)
        collected.append(document)
    return collected


def _rank_documents(plan: _QueryPlan, documents: list[_ScoredDocument]) -> list[_ScoredDocument]:
    """Score every candidate with weighted BM25 plus phrase and header-bigram bonuses."""
    if not documents:
        return []
    average_length = sum(document.length for document in documents) / len(documents)
    if average_length <= 0:
        average_length = 1.0
    document_frequency: Counter[str] = Counter()
    for document in documents:
        for term, _weight in plan.terms:
            if document.terms.get(term):
                document_frequency[term] += 1
    inverse_frequency = {
        term: math.log(1.0 + (len(documents) - hits + 0.5) / (hits + 0.5))
        for term, hits in document_frequency.items()
    }
    total_plan_terms = len({term for term, _weight in plan.terms})
    # A question typically names its company ("...for 3M?", "Amazon's ..."). Match
    # those tokens against the leading document-id line; year tokens alone are not
    # distinctive because several documents share the same fiscal year.
    identity_terms = {
        term for term, _weight in plan.terms
        if term and not (len(term) == 4 and term.isdigit())
    }
    for document in documents:
        score = 0.0
        matched_terms = 0
        first_line = document.text.split("\n", 1)[0].lower()
        first_tokens = set(_WORD_PATTERN.findall(first_line))
        if identity_terms & first_tokens:
            score += _DOCUMENT_IDENTITY_BONUS
        length_norm = _BM25_K1 * (1.0 - _BM25_B + _BM25_B * document.length / average_length)
        for term, weight in plan.terms:
            frequency = document.terms.get(term)
            if not frequency:
                continue
            matched_terms += 1
            score += weight * inverse_frequency[term] * frequency * (_BM25_K1 + 1.0) / (frequency + length_norm)
        if total_plan_terms:
            score += _COVERAGE_BONUS * matched_terms / total_plan_terms
        for bigram in plan.bigrams:
            first, second = bigram
            if document.bigrams.get(bigram):
                score += _BIGRAM_BONUS * max(2.0, inverse_frequency.get(first, 0.0) + inverse_frequency.get(second, 0.0))
            if document.header.get(bigram):
                score += _HEADER_BONUS
        for header in plan.concept_header_bigrams:
            if document.header.get(header):
                score += _HEADER_CONCEPT_BONUS
        if plan.concept_header_bigrams and document.is_primary_statement:
            score += _STATEMENT_PAGE_BONUS
        document.score = score
    return sorted(documents, key=lambda document: (-document.score, document.key))


async def _entry_source_ids(memory: Any, document: _ScoredDocument) -> tuple[str, ...]:
    """Resolve the exact Source identities behind one entry citation."""
    from powercontext.builtin.artifacts.memory.models import MemoryCitation
    from powercontext.builtin.runtime.models import GetMemoryEntryRequest

    try:
        record = await memory.get(
            GetMemoryEntryRequest(
                citation=MemoryCitation(
                    memory_ref=document.memory_ref,
                    entry_id=document.key[0],
                    entry_version_id=document.key[1],
                )
            )
        )
    except Exception:  # noqa: BLE001 - an unreadable entry simply yields no citation
        return ()
    return tuple(reference.source_id for reference in record.entry.sources)


async def search_pdf_pages(memory: Any, question: str, limit: int = 10) -> tuple[Any, ...]:
    """Return up to ``limit`` distinct evidence-page citations for one question.

    The scoped Memory is queried through its public search and listing APIs; the
    collected page texts are then re-ranked offline with a deterministic lexical
    BM25 model so no external service is involved.
    """
    from powercontext.builtin.artifacts.memory.models import MemoryCitation

    if limit < 1 or not isinstance(question, str) or not question.strip():
        return ()
    plan = _build_query_plan(question)
    documents = await _collect_documents(memory, question, limit)
    ranked = _rank_documents(plan, documents)
    citations: list[Any] = []
    claimed: set[str] = set()
    for document in ranked:
        source_ids = document.source_ids or await _entry_source_ids(memory, document)
        if not source_ids or claimed.intersection(source_ids):
            continue
        claimed.update(source_ids)
        citations.append(
            MemoryCitation(
                memory_ref=document.memory_ref,
                entry_id=document.key[0],
                entry_version_id=document.key[1],
            )
        )
        if len(citations) >= limit:
            break
    return tuple(citations)
