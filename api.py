import asyncio
import json
import logging
import re
from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from nemoguardrails.rails.llm.options import RailType
from pydantic import BaseModel, Field

from hybrid_retrieval import (
    BASE_DIR,
    HybridRetriever,
    ResponseFormatter,
    initialize_nemo_guardrails,
)
from lcel_chain import create_meridian_rag_chain
from prompt_injection_checker import PromptInjectionChecker
from redact_manager import RedactManager
from reranker import MeridianReranker
from security_validator import SecurityValidator

from validators import NoUnsupportedPromises
from guardrails.validator_base import FailResult

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Meridian Air GenAI Assistant API",
    description="API for the Meridian Air passenger policy assistant",
    version="1.0.0",
)


class AskRequest(BaseModel):
    question: str = Field(
        ..., min_length=1, max_length=2000,
        description="Passenger question about Meridian Air policies",
    )


class SourceReference(BaseModel):
    document_name: str | None = None
    document_id: str | None = None
    section: str | None = None
    subsection: str | None = None
    page_number: int | None = None
    chunk_id: str | int | None = None
    title: str | None = None


class AskResponse(BaseModel):
    answer: str
    sources: list[SourceReference]
    intent: str


class MeridianAssistantService:
    """Loads project components once and processes API questions."""

    def __init__(self):
        config_path = BASE_DIR / "config.yaml"
        if not config_path.exists():
            raise FileNotFoundError("config.yaml was not found")
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

        self.retriever = HybridRetriever(config)
        self.redactor = RedactManager()
        self.injection_checker = PromptInjectionChecker()
        self.security_validator = SecurityValidator()
        self.nemo_rails, self.nemo_actions = initialize_nemo_guardrails()
        self.answer_validator = NoUnsupportedPromises()
        self.formatter = ResponseFormatter()
        self.reranker = MeridianReranker(
            top_k=3,
            reference_date=date(2026, 11, 18),
        )
        self.rag_chain = create_meridian_rag_chain()

    @staticmethod
    def _source_references(documents: list[Any]) -> list[dict[str, Any]]:
        """Return only provenance fields actually present in chunk metadata."""
        references = []
        seen = set()
        for doc in documents:
            meta = doc.metadata or {}
            ref = {
                "document_name": meta.get("document_name") or meta.get("source"),
                "document_id": meta.get("document_id"),
                "section": meta.get("section"),
                "subsection": meta.get("subsection"),
                "page_number": meta.get("page_number") or meta.get("page"),
                "chunk_id": meta.get("chunk_index"),
                "title": meta.get("title"),
            }
            # Remove absent values; do not invent PDF page numbers.
            ref = {key: value for key, value in ref.items() if value is not None and value != ""}
            key = json.dumps(ref, sort_keys=True, default=str)
            if ref and key not in seen:
                seen.add(key)
                references.append(ref)
        return references

    def answer_question(self, question: str) -> dict[str, Any]:
        question = question.strip()
        input_result = self.security_validator.validate_input(question)
        if not input_result["is_valid"]:
            raise HTTPException(status_code=422, detail={
                "code": "INVALID_INPUT", "message": input_result["reason"]
            })

        injection_result = self.injection_checker.check_text(question)
        if injection_result["is_suspicious"]:
            raise HTTPException(status_code=400, detail={
                "code": "PROMPT_INJECTION_DETECTED",
                "message": "The request contains suspicious instructions.",
            })

        try:
            nemo_result = asyncio.run(self.nemo_rails.check_async(
                [{"role": "user", "content": question}],
                rail_types=[RailType.INPUT],
            ))
        except Exception as exc:
            logger.exception("NeMo input validation failed")
            raise HTTPException(status_code=503, detail={
                "code": "INPUT_GUARDRAIL_UNAVAILABLE",
                "message": "The request could not be securely validated. Please retry.",
            }) from exc

        nemo_status = getattr(nemo_result, "status", None)
        nemo_status_value = getattr(nemo_status, "value", nemo_status)
        if str(nemo_status_value).lower() == "blocked":
            raise HTTPException(status_code=400, detail={
                "code": "REQUEST_BLOCKED",
                "message": getattr(nemo_result, "content", None)
                    or "The request was blocked by the safety checks.",
            })

        try:
            safe_question = self.redactor.pii_filter(question)
            intent, documents = self.retriever.retrieve(safe_question)

            safe_documents = []
            for document in documents:
                check = self.injection_checker.check_text(document.page_content)
                nemo_flagged = self.nemo_actions.contains_indirect_prompt_injection([document])
                if not check["is_suspicious"] and not nemo_flagged:
                    safe_documents.append(document)
                else:
                    logger.warning("Excluded suspicious retrieved document")
            documents = safe_documents

            if intent == "complaint" or not documents:
                answer = self.formatter.format_response(intent, documents)
                final_documents = documents
            else:
                final_documents, _latency_ms = self.reranker.rerank(safe_question, documents)

                # Preserve the existing project's baggage-table handling.
                if intent == "baggage":
                    included_titles = {
                        str(doc.metadata.get("title", "")).strip().lower()
                        for doc in final_documents
                    }
                    for document in documents:
                        title = str(document.metadata.get("title", "")).strip().lower()
                        if title in {"table_8", "table_9", "table_13"} and title not in included_titles:
                            final_documents.append(document)
                            included_titles.add(title)

                answer = self.rag_chain.invoke({
                    "docs": final_documents,
                    "question": safe_question,
                })

            answer = str(answer)
            output_result = self.security_validator.validate_output(answer)
            if not output_result["is_valid"]:
                answer = (
                    "I cannot confirm or promise specific refund or compensation amounts. "
                    "Please contact Meridian Air's official case-handling channel for assistance."
                )

            answer = self.redactor.pii_filter(answer)
            privacy_check = asyncio.run(
                self.nemo_actions.check_generated_answer_privacy(
                    context={"generated_answer": answer}
                )
            )
            if privacy_check:
                answer = (
                    "I'm sorry, I can't provide this response because it may contain "
                    "confidential passenger information. Please contact Meridian Air support."
                )

            validation_result = self.answer_validator.validate(answer, metadata={})
            if isinstance(validation_result, FailResult):
                answer = (
                    "I can't confirm refund or compensation eligibility from the available "
                    "policy information. Please refer this case to Meridian Air's team."
                )

            return {
                "answer": answer,
                "sources": self._source_references(final_documents),
                "intent": intent,
            }
        except HTTPException:
            raise
        except Exception as exc:
            logger.exception("Question processing failed")
            raise HTTPException(status_code=500, detail={
                "code": "QUESTION_PROCESSING_FAILED",
                "message": "The assistant could not process this question. Please try again.",
            }) from exc


_service: MeridianAssistantService | None = None


@app.on_event("startup")
def load_assistant() -> None:
    global _service
    _service = MeridianAssistantService()
    logger.info("Meridian Air assistant initialized")


def get_service() -> MeridianAssistantService:
    if _service is None:
        raise HTTPException(status_code=503, detail={
            "code": "SERVICE_NOT_READY", "message": "Assistant is still initializing."
        })
    return _service


@app.get("/health")
def health():
    return {"status": "ok", "service": "meridian-air-assistant"}


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    return get_service().answer_question(request.question)


@app.post("/ask/stream")
def ask_stream(request: AskRequest):
    """Return a validated answer as server-sent events.

    The existing pipeline is fully executed and safety-checked before chunks are
    sent. This preserves the current output-guardrail behaviour. It is chunked
    HTTP delivery, not token-by-token LLM generation streaming.
    """
    result = get_service().answer_question(request.question)

    def event_generator():
        yield "data: " + json.dumps({"type": "sources", "sources": result["sources"]}) + "\n\n"
        answer = result["answer"]
        for start in range(0, len(answer), 40):
            yield "data: " + json.dumps({
                "type": "chunk", "content": answer[start:start + 40]
            }, ensure_ascii=False) + "\n\n"
        yield "data: " + json.dumps({"type": "done"}) + "\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
