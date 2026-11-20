
import asyncio
import importlib.util
import json
import logging
import re
import sys
from datetime import date
from pathlib import Path

import yaml
from langchain_core.documents import Document

try:
    from rich.console import Console
    from rich.markdown import Markdown
    from rich.panel import Panel

    console = Console(highlight=False)
except ImportError:
    Console = None
    Markdown = None
    Panel = None
    console = None

from storagemanager import StoreManager
from retrievemanager import RetrieveManager
from reranker import MeridianReranker
from lcel_chain import create_meridian_rag_chain
from redact_manager import RedactManager
from prompt_injection_checker import PromptInjectionChecker
from security_validator import SecurityValidator

from nemoguardrails import LLMRails, RailsConfig
from nemoguardrails.rails.llm.options import RailType


BASE_DIR = Path(__file__).resolve().parent

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


# ==========================================================
# GUARDRAILS AI: VALIDATOR IMPORT
# ==========================================================

GUARDRAILS_AI_DIR = BASE_DIR / "guardrails" / "guardrails_ai"

if str(GUARDRAILS_AI_DIR) not in sys.path:
    sys.path.insert(0, str(GUARDRAILS_AI_DIR))

from validators import NoUnsupportedPromises
from guardrails.validator_base import FailResult


# ==========================================================
# NEMO GUARDRAILS: INITIALIZATION
# ==========================================================

GUARDRAILS_DIR = BASE_DIR / "guardrails" / "nemo"


def initialize_nemo_guardrails():

    config_path = GUARDRAILS_DIR / "config.yml"
    actions_path = GUARDRAILS_DIR / "actions.py"

    if not config_path.exists():
        raise FileNotFoundError(
            f"NeMo configuration not found: {config_path}"
        )

    if not actions_path.exists():
        raise FileNotFoundError(
            f"NeMo actions file not found: {actions_path}"
        )

    nemo_config = RailsConfig.from_path(str(GUARDRAILS_DIR))

    action_spec = importlib.util.spec_from_file_location(
        "meridian_nemo_actions",
        actions_path
    )

    if action_spec is None or action_spec.loader is None:
        raise ImportError("Could not load NeMo custom actions.")

    action_module = importlib.util.module_from_spec(action_spec)
    action_spec.loader.exec_module(action_module)

    nemo_rails = LLMRails(nemo_config)

    nemo_rails.register_action(action_module.check_passenger_privacy)
    nemo_rails.register_action(action_module.check_prompt_injection)
    nemo_rails.register_action(
        action_module.check_generated_answer_privacy
    )

    logger.info("NeMo Guardrails initialized.")
    logger.info(
        "Registered passenger privacy, prompt-injection, "
        "and generated-answer privacy actions."
    )

    return nemo_rails, action_module


# ==========================================================
# 1. TEXT PROCESSING
# ==========================================================

def normalize(text):

    return " ".join(
        re.findall(r"[a-z0-9]+", text.lower())
    )


# ==========================================================
# 2. QUESTION UNDERSTANDING
# ==========================================================

class QuestionUnderstanding:

    def __init__(self):

        self.intent_patterns = {

            "all_policies": [
                "what are the policies",
                "what are your policies",
                "what are all the policies",
                "what are the airline policies",
                "list all policies",
                "list the policies",
                "list policies",
                "tell me about the policies",
                "tell me about your policies",
                "explain the policies",
                "explain your policies",
                "show me the policies",
                "what policies do you have",
                "what are the main policies",
                "all airline policies",
                "all policies",
                "main policies",
                "policy overview",
                "eight policies"
            ],

            "company": [
                "tell me about the company",
                "tell me about company",
                "company background",
                "company information",
                "about meridian air",
                "what is meridian air",
                "who is meridian air",
                "airline background",
                "company profile",
                "about your airline"
            ],

            "business_problem": [
                "business problem",
                "what is the business problem",
                "what problem are you solving",
                "what is the problem",
                "problem faced by the company",
                "customer service problem",
                "operational problem",
                "challenges faced by the airline",
                "what are the challenges"
            ],

            "objectives": [
                "business objectives",
                "expected business outcomes",
                "what are your objectives",
                "what are the goals",
                "company goals",
                "expected outcomes",
                "what do you want to achieve"
            ],

            "fare": [
                "fare policy",
                "fare conditions",
                "ticket conditions",
                "ticket restrictions",
                "fare types",
                "ticket price conditions"
            ],

            "baggage": [
                "baggage policy",
                "baggage allowance",
                "baggage rules",
                "luggage policy",
                "checked baggage",
                "cabin baggage",
                "extra baggage"
            ],

            "rebooking": [
                "ticket changes",
                "change my ticket",
                "change flight",
                "rebooking policy",
                "reschedule flight",
                "flight change conditions"
            ],

            "refund": [
                "refund policy",
                "ticket refund",
                "get my money back",
                "return my ticket amount",
                "cancel my ticket",
                "cancellation policy",
                "refund conditions"
            ],

            "flight_disruption": [
                "flight delay",
                "delayed flight",
                "flight cancellation",
                "cancelled flight",
                "missed flight",
                "flight disruption",
                "irregular operations"
            ],

            "passenger_rights": [
                "passenger rights",
                "passenger compensation",
                "compensation policy",
                "passenger protection",
                "my travel rights"
            ],

            "special_assistance": [
                "special assistance",
                "wheelchair assistance",
                "disabled passenger",
                "reduced mobility",
                "special needs"
            ],

            "airport_transfer": [
                "airport transfer",
                "connecting flight",
                "missed connection",
                "airport connection",
                "transfer between terminals",
                "connection time"
            ]
        }

        self.policy_sections = {
            "fare": "2",
            "baggage": "3",
            "rebooking": "3",
            "refund": "2",
            "flight_disruption": "5",
            "passenger_rights": "6",
            "special_assistance": "3",
            "airport_transfer": "7"
        }

    def detect_intent(self, question):

        query = normalize(question)

        # Check specific policy questions first.

        for intent in self.policy_sections:

            for phrase in self.intent_patterns[intent]:

                if normalize(phrase) in query:
                    return intent

        # Check complaints.

        complaint_words = [
            "neglect",
            "ignored",
            "violated",
            "failed to follow",
            "not following",
            "did not follow",
            "didn't follow",
            "not followed",
            "you broke",
            "you did not follow"
        ]

        if any(word in query for word in complaint_words):

            for intent in self.policy_sections:

                for phrase in self.intent_patterns[intent]:

                    if normalize(phrase) in query:
                        return "complaint_" + intent

            return "complaint"

        # Check all policies.

        for phrase in self.intent_patterns["all_policies"]:

            if normalize(phrase) in query:
                return "all_policies"

        # Check other main sections.

        for intent in [
            "company",
            "business_problem",
            "objectives"
        ]:

            for phrase in self.intent_patterns[intent]:

                if normalize(phrase) in query:
                    return intent

        return "general"


# ==========================================================
# 3. HYBRID RETRIEVER
# ==========================================================

class HybridRetriever:

    def __init__(self, config):

        chunks_path = BASE_DIR / "output" / "chunks.json"

        if not chunks_path.exists():

            raise FileNotFoundError(
                "output/chunks.json not found. "
                "Run your existing main.py first."
            )

        raw_chunks = json.loads(
            chunks_path.read_text(encoding="utf-8")
        )

        self.documents = [

            Document(
                page_content=item["page_content"],
                metadata=item.get("metadata", {})
            )

            for item in raw_chunks
        ]

        if not self.documents:

            raise ValueError(
                "No document chunks found."
            )

        self.store = StoreManager(config)

        self.retrieve_manager = RetrieveManager()

        self.question_understanding = QuestionUnderstanding()

        logger.info(
            "Loaded %s document chunks.",
            len(self.documents)
        )

        logger.info(
            "ChromaDB contains %s vectors.",
            self.store.get_collection_count()
        )

    # ------------------------------------------------------
    # Get section number from metadata or heading
    # ------------------------------------------------------

    @staticmethod
    def get_section_number(document):

        metadata = document.metadata

        # Check the most specific heading first.

        for key in [
            "subsection",
            "section",
            "title"
        ]:

            value = str(metadata.get(key, ""))

            match = re.search(
                r"^\s*(?:#{1,6}\s*)?(\d{1,2}(?:\.\d{1,2})*)\b",
                value
            )

            if match:
                return match.group(1)

        # Check the first heading in the chunk.
        # Do not search arbitrary numbers throughout the content.

        for line in document.page_content.splitlines()[:5]:

            match = re.match(
                r"^\s*#{1,6}\s*(\d{1,2}(?:\.\d{1,2})*)\b",
                line
            )

            if match:
                return match.group(1)

        return None

    # ------------------------------------------------------
    # Retrieve exact subsection documents
    # ------------------------------------------------------

    def get_section_documents(self, section_number):

        selected = []

        for document in self.documents:

            document_section = self.get_section_number(
                document
            )

            if document_section == section_number:

                selected.append(document)

        return selected

    # ------------------------------------------------------
    # Retrieve complete main section
    # ------------------------------------------------------

    def get_main_section_documents(self, main_section):

        selected = []

        for document in self.documents:

            document_section = self.get_section_number(
                document
            )

            if not document_section:
                continue

            main_number = document_section.split(".")[0]

            if main_number == main_section:

                selected.append(document)

        return selected

    # ------------------------------------------------------
    # General hybrid retrieval
    # ------------------------------------------------------

    def hybrid_search(self, question):

        return self.retrieve_manager.query_hybrid(
            query=question,
            documents=self.documents,
            vector_store=self.store.vector_store,
            n_results=10,
            candidate_results=10
        )

    # ------------------------------------------------------
    # Main retrieval decision
    # ------------------------------------------------------

    @staticmethod
    def is_test_fixture(document):
        """
        Exclude explicitly labelled prompt-injection test fixtures.
        These are evaluation examples, not operational airline policy.
        """
        content = document.page_content.lower()
        fixture_markers = (
            "injection test fixture",
            "p-01 - injection",
            "p-02 - injection",
            "p-03 - injection",
        )
        return any(marker in content for marker in fixture_markers)

    @staticmethod
    def _deduplicate_documents(documents):
        unique = []
        seen = set()

        for document in documents:
            content = document.page_content.strip()
            key = (
                str(document.metadata.get("title", "")),
                content
            )
            if content and key not in seen:
                seen.add(key)
                unique.append(document)

        return unique

    def _direct_policy_documents(self, intent):
        """
        Select known policy sections and extracted tables whose identifiers
        are present in the current Meridian Air corpus.
        """
        selected = []

        for document in self.documents:
            metadata = document.metadata
            section = str(metadata.get("section", "")).lower()
            title = str(metadata.get("title", "")).lower()
            content = document.page_content.lower()

            if self.is_test_fixture(document):
                continue

            if intent == "special_assistance":
                if (
                    section.startswith("bp-07")
                    or "other policies in brief" in section
                    or title in {"table_12"}
                    or "wheelchair and mobility-aid assistance" in content
                ):
                    selected.append(document)

            elif intent in {"flight_disruption", "passenger_rights"}:
                if (
                    section.startswith("5.")
                    or section.startswith("6.")
                    or title in {"table_14", "table_16", "table_17", "table_18"}
                ):
                    selected.append(document)

            elif intent == "airport_transfer":
                if (
                    section.startswith("7.")
                    or section.startswith("8.")
                    or section.startswith("ct-01")
                    or title == "table_15"
                ):
                    selected.append(document)

            elif intent == "all_policies":
                if (
                    section.startswith("3.")
                    or section.startswith("4.")
                    or title in {"table_7", "table_8", "table_9", "table_10",
                                 "table_11", "table_12", "table_13"}
                ):
                    selected.append(document)

        return selected

    def retrieve(self, question):

        intent = self.question_understanding.detect_intent(question)

        logger.info("Detected question intent: %s", intent)

        if intent == "complaint":
            return intent, []

        # Baggage allowances are represented by three source tables.
        if intent == "baggage":
            results = [
                document for document in self.documents
                if str(document.metadata.get("title", "")).strip().lower()
                in {"table_8", "table_9", "table_13"}
                and not self.is_test_fixture(document)
            ]
            logger.info(
                "Retrieved baggage tables: %s",
                [doc.metadata.get("title", "") for doc in results]
            )
            return intent, results

        # For policy topics whose records use document IDs (BP-07, CT-01,
        # IM-01, etc.), numeric section-only lookup is insufficient.
        direct_documents = self._direct_policy_documents(intent)

        # Hybrid search supplies relevant fare-specific and general records.
        hybrid_documents = self.hybrid_search(question)

        # Do not pass explicitly labelled injection fixtures to the model.
        safe_candidates = [
            document
            for document in direct_documents + hybrid_documents
            if not self.is_test_fixture(document)
        ]

        results = self._deduplicate_documents(safe_candidates)

        logger.info(
            "Retrieved %s candidate document(s) for intent '%s': %s",
            len(results),
            intent,
            [
                document.metadata.get("title")
                or document.metadata.get("section")
                or "untitled"
                for document in results
            ],
        )

        return intent, results


def display_answer(answer):
    """Display the answer inside a Rich panel with Markdown rendering."""
    text = str(answer)

    # Remove ANSI escape sequences supplied by untrusted model output.
    text = re.sub(
        r"\x1B(?:\][^\x07]*(?:\x07|\x1B\\)|\[[0-?]*[ -/]*[@-~]|[@-_])"
        r"|\x9B[0-?]*[ -/]*[@-~]",
        "",
        text,
    )

    # Remove terminal control characters while preserving newlines and tabs.
    text = "".join(
        char
        for char in text
        if char in "\n\t" or not (ord(char) < 32 or 127 <= ord(char) <= 159)
    ).strip()

    if console is not None and Markdown is not None and Panel is not None:
        console.print(
            Panel(
                Markdown(text),
                title="[bold cyan]Meridian Air Assistant[/bold cyan]",
                border_style="cyan",
                padding=(1, 2),
                expand=False,
            )
        )
    else:
        print("\n" + "=" * 60)
        print("MERIDIAN AIR ASSISTANT")
        print("=" * 60)
        print(text)
        print("=" * 60)


# ==========================================================
# 4. RESPONSE FORMATTER
# ==========================================================

class ResponseFormatter:

    @staticmethod
    def format_response(intent, documents):

        if intent == "complaint":

            return (
                "I understand that you are reporting a possible "
                "failure to follow Meridian Air's policies. "
                "Could you clarify which policy or issue you "
                "experienced: baggage, refund, ticket changes, "
                "flight disruption, passenger rights, special "
                "assistance or airport transfers?"
            )

        if not documents:

            return (
                "I could not find the requested information "
                "in the Meridian Air document. Please try "
                "another question."
            )

        output = []

        for index, document in enumerate(
            documents,
            start=1
        ):

            content = document.page_content.strip()

            if not content:
                continue

            metadata = document.metadata

            title = metadata.get("title", "")
            section = metadata.get("section", "")
            subsection = metadata.get("subsection", "")

            heading = " | ".join(
                str(value)
                for value in [
                    title,
                    section,
                    subsection
                ]
                if value
            )

            if heading:

                output.append(
                    f"\n[{index}] {heading}\n"
                )

            else:

                output.append(
                    f"\n[{index}] Relevant document content:\n"
                )

            output.append(content)

        if not output:

            return "No readable document content was retrieved."

        return "\n".join(output)


# ==========================================================
# 5. MAIN APPLICATION
# ==========================================================

def main():

    config_path = BASE_DIR / "config.yaml"

    if not config_path.exists():

        raise FileNotFoundError(
            "config.yaml not found."
        )

    config = yaml.safe_load(
        config_path.read_text(encoding="utf-8")
    )

    retriever = HybridRetriever(config)

    # Initialize PII redaction manager once.

    redactor = RedactManager()

    # Initialize the new ZIP's existing security components.

    injection_checker = PromptInjectionChecker()

    security_validator = SecurityValidator()

    # Initialize NeMo Guardrails and Guardrails AI.
    nemo_rails, nemo_actions = initialize_nemo_guardrails()
    answer_validator = NoUnsupportedPromises()

    formatter = ResponseFormatter()

    reranker = MeridianReranker(
        top_k=3,
        reference_date=date(2026, 11, 18)
    )

    rag_chain = create_meridian_rag_chain()

    print("\n" + "=" * 60)
    print("       MERIDIAN AIR CUSTOMER SERVICE ASSISTANT")
    print("=" * 60)

    print("\nAsk questions about Meridian Air.")
    print("Type 'exit' to stop.")

    while True:

        question = input("\nYou: ").strip()

        if question.lower() in ["exit", "quit"]:

            print(
                "\nThank you for using Meridian Air Assistant."
            )

            break

        # STEP 1: INPUT VALIDATION

        input_result = security_validator.validate_input(
            question
        )

        if not input_result["is_valid"]:

            print(
                "\nInvalid input:",
                input_result["reason"]
            )

            continue

        # STEP 2: PROMPT-INJECTION DETECTION ON USER INPUT

        injection_result = injection_checker.check_text(
            question
        )

        if injection_result["is_suspicious"]:

            logger.warning(
                "Prompt injection detected in user question. Patterns: %s",
                injection_result["matched_patterns"]
            )

            print(
                "\nYour question contains suspicious instructions. "
                "Please enter a normal airline-related question."
            )

            continue

        # STEP 3: NE MO INPUT RAILS
        try:
            nemo_result = asyncio.run(
                nemo_rails.check_async(
                    [{"role": "user", "content": question}],
                    rail_types=[RailType.INPUT],
                )
            )
        except Exception:
            logger.exception("NeMo input validation failed.")
            print(
                "\nMeridian Air Assistant could not validate the "
                "request securely. Please try again."
            )
            continue

        nemo_status = getattr(nemo_result, "status", None)
        nemo_status_value = getattr(nemo_status, "value", nemo_status)

        if str(nemo_status_value).lower() == "blocked":
            blocked_message = getattr(nemo_result, "content", "")
            print("\nMeridian Air Assistant:")
            print("-" * 60)
            print(
                blocked_message
                or "I'm sorry, I can't help with that request."
            )
            print("-" * 60)
            continue

        try:

            # STEP 4: REDACT PII FROM THE USER QUESTION

            safe_question = redactor.pii_filter(question)

            logger.info("User question processed for PII.")

            # STEP 5: RETRIEVE DOCUMENTS

            intent, documents = retriever.retrieve(
                safe_question
            )

            # STEP 6: CHECK RETRIEVED DOCUMENTS FOR INJECTION
            # Apply both the new ZIP's checker and the NeMo helper.

            safe_documents = []

            for document in documents:

                checker_result = injection_checker.check_text(
                    document.page_content
                )

                nemo_flagged = nemo_actions.contains_indirect_prompt_injection(
                    [document]
                )

                if checker_result["is_suspicious"] or nemo_flagged:

                    logger.warning(
                        "Suspicious retrieved document excluded. "
                        "PromptInjectionChecker patterns: %s; "
                        "NeMo flagged: %s",
                        checker_result.get("matched_patterns", []),
                        nemo_flagged
                    )

                else:

                    safe_documents.append(document)

            documents = safe_documents

            # STEP 7: RERANK & GENERATE RESPONSE VIA LCEL

            if intent == "complaint" or not documents:

                answer = formatter.format_response(
                    intent,
                    documents
                )

            else:

                reranked_docs, _ = reranker.rerank(
                    safe_question,
                    documents
                )

                # Baggage policy is split across three extracted source tables.
                # FlashRank can filter a valid table below its score threshold,
                # especially for negative/edge-case wording such as "free
                # unlimited baggage". Keep every safely retrieved baggage
                # source in the final context so the LLM can compare the actual
                # checked-baggage tables and cabin-baggage rules.
                if intent == "baggage":
                    included_titles = {
                        str(doc.metadata.get("title", "")).strip().lower()
                        for doc in reranked_docs
                    }

                    for document in documents:
                        title = str(
                            document.metadata.get("title", "")
                        ).strip().lower()

                        if (
                            title in {"table_8", "table_9", "table_13"}
                            and title not in included_titles
                        ):
                            reranked_docs.append(document)
                            included_titles.add(title)

                    logger.info(
                        "Baggage documents passed to Gemini after reranking: %s",
                        [
                            document.metadata.get("title", "")
                            for document in reranked_docs
                        ]
                    )

                answer = rag_chain.invoke({
                    "docs": reranked_docs,
                    "question": safe_question
                })

            # STEP 8: CONVERT MODEL RESPONSE TO STRING

            answer = str(answer)

            # STEP 9: OUTPUT VALIDATION USING THE NEW ZIP'S VALIDATOR

            output_result = security_validator.validate_output(
                answer
            )

            if not output_result["is_valid"]:

                logger.warning(
                    "Unsafe response blocked: %s",
                    output_result["reason"]
                )

                answer = (
                    "I cannot confirm or promise specific refund "
                    "or compensation amounts. Please contact "
                    "Meridian Air's official case-handling channel "
                    "for assistance."
                )

            # STEP 10: REDACT PII FROM THE GENERATED ANSWER

            answer = redactor.pii_filter(answer)

            # STEP 11: NE MO GENERATED-ANSWER PRIVACY CHECK

            privacy_check = asyncio.run(
                nemo_actions.check_generated_answer_privacy(
                    context={"generated_answer": answer}
                )
            )

            if privacy_check:

                logger.warning(
                    "NeMo blocked generated answer containing "
                    "potentially confidential passenger information."
                )

                answer = (
                    "I'm sorry, I can't provide this response because "
                    "it may contain confidential passenger information. "
                    "Please contact Meridian Air's support team."
                )

            else:

                logger.info(
                    "NeMo approved generated-answer privacy check."
                )

            # STEP 12: GUARDRAILS AI UNSUPPORTED-PROMISE VALIDATION

            validation_result = answer_validator.validate(
                answer,
                metadata={},
            )

            if isinstance(validation_result, FailResult):

                logger.warning(
                    "Guardrails AI blocked an unsupported refund "
                    "or compensation promise."
                )

                answer = (
                    "I can't confirm refund or compensation eligibility "
                    "from the available policy information. Please refer "
                    "this case to Meridian Air's case-handling team for review."
                )

            else:

                logger.info(
                    "Guardrails AI approved the generated answer."
                )

            # STEP 13: DISPLAY THE ANSWER

            display_answer(answer)

        except Exception as error:

            logger.exception(
                "Question processing failed."
            )

            print(
                "\nAn error occurred:",
                error
            )


if __name__ == "__main__":

    main()
