import re
import time
from datetime import date, datetime
from typing import Any, Dict, List, Optional, Tuple, Union
from langchain_core.documents import Document
from flashrank import Ranker, RerankRequest


METADATA_ALIASES: Dict[str, List[str]] = {
    "document_id": ["document_id", "doc_id", "id", "policy_id", "manual", "policy"],
    "document_name": ["document_name", "source", "title", "file_name", "filename"],
    "section": ["section", "subsection", "page_number", "page", "page_or_section"],
    "version": ["version", "ver", "doc_version", "revision"],
    "effective_date": ["effective_date", "effective_from", "effective", "date"],
    "effective_to": ["effective_to", "valid_to", "expiry_date"],
    "status": ["status", "doc_status"],
    "is_superseded": ["is_superseded", "superseded", "deprecated", "is_outdated"],
    "content_type": ["document_type", "content_type", "type", "chunk_type", "modality"],
}

DOC_ID_REGEX = re.compile(r"\b((?:FC|BP|RR|IM|MAP|CT)-\d{2,3})\b", re.IGNORECASE)
DATE_REGEX = re.compile(r"\b(20\d{2}-\d{2}-\d{2})\b")


def _extract_meta_value(metadata: Dict[str, Any], canonical_key: str, default: Any = None) -> Any:
    for alias in METADATA_ALIASES.get(canonical_key, [canonical_key]):
        if alias in metadata and metadata[alias] is not None:
            return metadata[alias]
    return default


def _parse_iso_date(value: Any) -> Optional[date]:
    if not value:
        return None
    match = DATE_REGEX.search(str(value))
    if not match:
        return None
    try:
        return datetime.strptime(match.group(1), "%Y-%m-%d").date()
    except ValueError:
        return None


def _infer_temporal_status(
    page_content: str,
    raw_meta: Dict[str, Any],
    reference_date: date,
) -> Tuple[str, str, str, bool]:
    """
    Extracts document_id, effective_from, effective_to, and inactive/superseded status
    from either explicit metadata or inline table/text content (e.g., FC-007 expired
    2026-08-31 or FC-029 future-dated 2027-01-01 in airline_case_study.pdf).
    """
    doc_id = str(_extract_meta_value(raw_meta, "document_id", "")).upper()
    if not doc_id:
        id_match = DOC_ID_REGEX.search(page_content)
        doc_id = id_match.group(1).upper() if id_match else "DOC-GENERAL"

    eff_from_str = str(_extract_meta_value(raw_meta, "effective_date", ""))
    eff_to_str = str(_extract_meta_value(raw_meta, "effective_to", ""))

    dates_in_text = DATE_REGEX.findall(page_content)
    if not eff_from_str and dates_in_text:
        eff_from_str = dates_in_text[0]
    if not eff_to_str and len(dates_in_text) >= 2:
        eff_to_str = dates_in_text[1]

    eff_from_dt = _parse_iso_date(eff_from_str)
    eff_to_dt = _parse_iso_date(eff_to_str)

    explicit_superseded = _extract_meta_value(raw_meta, "is_superseded", None)
    status_meta = str(_extract_meta_value(raw_meta, "status", "")).lower()

    is_inactive = False
    if explicit_superseded is True or "superseded" in status_meta or "expired" in status_meta:
        is_inactive = True
    elif "future" in page_content.lower() or (eff_from_dt and eff_from_dt > reference_date):
        is_inactive = True
    elif eff_to_dt and eff_to_dt < reference_date:
        is_inactive = True

    return (
        doc_id,
        eff_from_str or "2026-01-01",
        eff_to_str or "none",
        is_inactive,
    )


def normalize_to_document(
    item: Union[Document, Tuple[Any, float], Dict[str, Any], str],
    reference_date: date,
) -> Document:
    """
    Accepts:
    - `list[Document]` from `HybridRetriever.retrieve(question)` -> `(intent, documents)`
      or `HybridRetriever.hybrid_search(question)`
    - `(Document, score)` tuples
    - Plain `dict` or `str` chunks
    """
    rrf_score: Optional[float] = None

    if isinstance(item, tuple):
        if len(item) != 2:
            raise ValueError(f"Expected (document, score) tuple, got length {len(item)}")
        inner_doc, raw_score = item
        try:
            rrf_score = round(float(raw_score), 6)
        except (TypeError, ValueError):
            rrf_score = None
        doc_obj = normalize_to_document(inner_doc, reference_date)
        if rrf_score is not None:
            doc_obj.metadata["retrieval_score"] = rrf_score
        return doc_obj

    if isinstance(item, str):
        page_content = item
        raw_meta: Dict[str, Any] = {}
    elif isinstance(item, Document):
        page_content = item.page_content
        raw_meta = dict(item.metadata or {})
    elif isinstance(item, dict):
        page_content = item.get("page_content") or item.get("text") or item.get("content") or ""
        raw_meta = dict(
            item.get("metadata")
            or {k: v for k, v in item.items() if k not in ("page_content", "text", "content")}
        )
    else:
        raise TypeError(f"Unsupported candidate chunk type: {type(item)}")

    doc_id, eff_from, eff_to, is_inactive = _infer_temporal_status(
        page_content, raw_meta, reference_date
    )

    section_parts = [
        str(raw_meta[k])
        for k in ("title", "section", "subsection")
        if k in raw_meta and raw_meta[k]
    ]
    resolved_section = (
        " > ".join(section_parts)
        if section_parts
        else str(_extract_meta_value(raw_meta, "section", "General Policy"))
    )

    normalized_meta = dict(raw_meta)
    normalized_meta["document_id"] = doc_id
    normalized_meta["document_name"] = str(
        _extract_meta_value(raw_meta, "document_name", "airline_case_study.pdf")
    )
    normalized_meta["section"] = resolved_section
    normalized_meta["version"] = str(_extract_meta_value(raw_meta, "version", "v1.0"))
    normalized_meta["effective_date"] = eff_from
    normalized_meta["effective_to"] = eff_to
    normalized_meta["is_superseded"] = is_inactive
    normalized_meta["status"] = "Inactive/Superseded" if is_inactive else "Current"
    normalized_meta["content_type"] = str(
        _extract_meta_value(raw_meta, "content_type", "text")
    )

    return Document(page_content=page_content, metadata=normalized_meta)


class MeridianReranker:
    """
    Member 4 – Reranking:
    - Accepts either `(intent, list[Document])` returned directly by Member 3's
      `HybridRetriever.retrieve(question)` OR `list[Document]` from `hybrid_search(question)`.
    - Reranks candidates using lightweight FlashRank ONNX (`ms-marco-MiniLM-L-12-v2`).
    - Enforces F3 & Date-Aware Precedence: active current documents strictly outrank
      expired (e.g., FC-007 ended 2026-08-31) and future-dated (e.g., FC-029 starts 2027-01-01) rules.
    - Preserves `source`, `title`, `section`, `subsection`, `chunk_index`, and `content_hash`.
    """

    def __init__(
        self,
        model_name: str = "ms-marco-MiniLM-L-12-v2",
        top_k: int = 5,
        min_score_threshold: float = 0.05,
        cache_dir: str = "./rerank_cache",
        reference_date: Optional[date] = None,
    ):
        self.ranker = Ranker(model_name=model_name, cache_dir=cache_dir)
        self.top_k = top_k
        self.min_score_threshold = min_score_threshold
        self.reference_date = reference_date or date.today()

    def rerank(
        self,
        query: str,
        candidate_docs: Union[
            Tuple[str, List[Document]],
            List[Union[Document, Tuple[Any, float], Dict[str, Any], str]],
        ],
        top_k: Optional[int] = None,
        reference_date: Optional[date] = None,
    ) -> Tuple[List[Document], float]:
        start = time.perf_counter()
        limit = top_k if top_k is not None else self.top_k
        ref_date = reference_date or self.reference_date

        # Unpack `(intent, documents)` if passed directly from `HybridRetriever.retrieve(question)`
        detected_intent: Optional[str] = None
        if (
            isinstance(candidate_docs, tuple)
            and len(candidate_docs) == 2
            and isinstance(candidate_docs[0], str)
            and isinstance(candidate_docs[1], list)
        ):
            detected_intent, docs_list = candidate_docs
        else:
            docs_list = list(candidate_docs)

        if not docs_list:
            return [], 0.0

        normalized_docs = [normalize_to_document(doc, ref_date) for doc in docs_list]
        if detected_intent:
            for doc in normalized_docs:
                doc.metadata["detected_intent"] = detected_intent

        passages = [
            {"id": idx, "text": doc.page_content, "meta": doc.metadata}
            for idx, doc in enumerate(normalized_docs)
        ]

        results = self.ranker.rerank(RerankRequest(query=query, passages=passages))

        scored_docs: List[Document] = []
        for item in results:
            score = float(item["score"])
            if score < self.min_score_threshold:
                continue
            doc = normalized_docs[item["id"]]
            meta = dict(doc.metadata)
            meta["rerank_score"] = round(score, 4)
            scored_docs.append(Document(page_content=doc.page_content, metadata=meta))

        if not scored_docs and results:
            for item in results[:limit]:
                doc = normalized_docs[item["id"]]
                meta = dict(doc.metadata)
                meta["rerank_score"] = round(float(item["score"]), 4)
                scored_docs.append(Document(page_content=doc.page_content, metadata=meta))

        # Active/Current documents strictly outrank expired/future-dated/superseded ones
        scored_docs.sort(
            key=lambda d: (
                not d.metadata["is_superseded"],
                d.metadata["rerank_score"],
                d.metadata.get("retrieval_score", 0.0),
                d.metadata["effective_date"],
            ),
            reverse=True,
        )

        temporal_query = any(
            kw in query.lower()
            for kw in ("expired", "superseded", "future", "2027", "old policy", "previous")
        )
        active_docs = [d for d in scored_docs if not d.metadata["is_superseded"]]

        if active_docs and not temporal_query:
            final_docs = active_docs[:limit]
        else:
            final_docs = scored_docs[:limit]

        latency_ms = round((time.perf_counter() - start) * 1000, 2)
        return final_docs, latency_ms