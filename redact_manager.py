
from presidio_analyzer import (
    AnalyzerEngine,
    Pattern,
    PatternRecognizer,
)
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig
from langchain_core.documents import Document


class RedactManager:

    def __init__(self):

        self.analyzer = AnalyzerEngine()
        self.anonymizer = AnonymizerEngine()

        self.entities = [
            "PERSON",
            "EMAIL_ADDRESS",
            "PHONE_NUMBER",
            "CREDIT_CARD",
            "IBAN_CODE",
            "IP_ADDRESS",
            "PASSPORT_NUMBER",
            "BOOKING_REFERENCE",
            "TICKET_NUMBER",
        ]

        self.operators = {
            entity: OperatorConfig(
                "replace",
                {"new_value": f"<{entity}>"}
            )
            for entity in self.entities
        }

        self._register_custom_recognizers()

    # 1. Register custom PII recognizers

    def _register_custom_recognizers(self):

        recognizers = [

            PatternRecognizer(
                supported_entity="PASSPORT_NUMBER",
                patterns=[
                    Pattern(
                        name="passport_pattern",
                        regex=r"\b[A-Z]{1,2}[0-9]{6,9}\b",
                        score=0.85
                    )
                ]
            ),

            PatternRecognizer(
                supported_entity="BOOKING_REFERENCE",
                patterns=[
                    Pattern(
                        name="booking_reference_pattern",
                        regex=(
                            r"\b(?=[A-Z0-9]{6}\b)"
                            r"(?=[A-Z0-9]*[A-Z])"
                            r"(?=[A-Z0-9]*\d)"
                            r"[A-Z0-9]{6}\b"
                        ),
                        score=0.85
                    )
                ]
            ),

            PatternRecognizer(
                supported_entity="TICKET_NUMBER",
                patterns=[
                    Pattern(
                        name="ticket_number_pattern",
                        regex=r"\b[0-9]{13}\b",
                        score=0.85
                    )
                ]
            ),

            PatternRecognizer(
                supported_entity="PHONE_NUMBER",
                patterns=[
                    Pattern(
                        name="international_phone_pattern",
                        regex=r"\+\d{1,3}(?:[\s.-]?\d){7,14}\b",
                        score=0.90
                    )
                ]
            ),

            PatternRecognizer(
                supported_entity="EMAIL_ADDRESS",
                patterns=[
                    Pattern(
                        name="custom_email_pattern",
                        regex=r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
                        score=0.95
                    )
                ]
            ),
        ]

        for recognizer in recognizers:
            self.analyzer.registry.add_recognizer(recognizer)

    # 2. Remove unsuitable PERSON detections

    @staticmethod
    def _filter_person_results(text, results):

        filtered_results = []

        ignored_person_labels = {
            "email",
            "phone",
            "passenger",
            "booking",
            "reference",
            "number",
            "passport",
        }

        for result in results:

            if result.entity_type == "PERSON":

                matched_text = text[
                    result.start:result.end
                ]

                # Ignore labels that are not actual person names.

                if matched_text.strip().lower() in ignored_person_labels:
                    continue

                # Avoid masking a large multi-line text section.

                if "\n" in matched_text:
                    continue

            filtered_results.append(result)

        return filtered_results

    # 3. Main PII redaction method

    def pii_filter(self, text: str) -> str:

        if not text:
            return text

        results = self.analyzer.analyze(
            text=text,
            language="en",
            entities=self.entities,
            score_threshold=0.70
        )

        results = self._filter_person_results(
            text,
            results
        )

        if not results:
            return text

        anonymized_result = self.anonymizer.anonymize(
            text=text,
            analyzer_results=results,
            operators=self.operators
        )

        return anonymized_result.text

    # 4. Redact LangChain Document content

    def redact_document(self, document: Document) -> Document:

        if not isinstance(document, Document):
            raise TypeError(
                "Input must be a LangChain Document."
            )

        redacted_content = self.pii_filter(
            document.page_content
        )

        return Document(
            page_content=redacted_content,
            metadata=dict(document.metadata)
        )

    # 5. Redact multiple LangChain Documents

    def redact_documents(
        self,
        documents: list[Document]
    ) -> list[Document]:

        return [
            self.redact_document(document)
            for document in documents
        ]
