from datetime import date
from typing import List, Tuple, Union
from langchain_core.documents import Document
from reranker import MeridianReranker


def get_case_study_test_cases() -> List[
    Tuple[str, str, Union[Tuple[str, List[Document]], List[Document]]]
]:
    """
    Ground-truth test cases simulating exact outputs returned by her `HybridRetriever`:
    - `(intent, list[Document])` from `HybridRetriever.retrieve(question)`
    - `list[Document]` from `HybridRetriever.hybrid_search(question)`
    """
    # 1. Section 4.2 Baggage Policy: Chennai -> London Economy Standard 25 kg suitcase
    q1 = "I booked an Economy Standard ticket from Chennai to London. Can I carry a 25 kg checked suitcase?"
    q1_candidates = (
        "baggage",
        [
            Document(
                page_content="Illustrative baggage allowance table: Chennai - Singapore | Economy Standard | Checked Baggage: 20 kg | Cabin Baggage: 7 kg.",
                metadata={
                    "source": "airline_case_study.pdf",
                    "section": "4. Airline Operations and Policy Areas",
                    "subsection": "4.2 Baggage Allowance Policy",
                    "document_type": "text",
                    "chunk_index": 7,
                    "chunk_id": "baggage_chennai_singapore",
                },
            ),
            Document(
                page_content="Illustrative baggage allowance table: Chennai - London | Economy Standard | Checked Baggage: 23 kg | Cabin Baggage: 7 kg. A 25 kg suitcase exceeds the 23 kg allowance and incurs excess baggage charges.",
                metadata={
                    "source": "airline_case_study.pdf",
                    "section": "4. Airline Operations and Policy Areas",
                    "subsection": "4.2 Baggage Allowance Policy",
                    "document_type": "text",
                    "chunk_index": 6,
                    "chunk_id": "baggage_chennai_london",
                },
            ),
        ],
    )

    # 2. Section 11.1 Date-Aware Hook: Expired FC-007 (ended 2026-08-31) & Future FC-029 (starts 2027-01-01)
    q2 = "What is the change fee and checked baggage allowance for Standard and Premium fares?"
    q2_candidates = [
        Document(
            page_content="FC-007 | Fare: Standard | Market: NAM | Effective From: 2026-02-01 | Effective To: 2026-08-31 | Change: $75 | Refund: 60% fare | Checked Bag: 1x23 kg.",
            metadata={
                "source": "airline_case_study.pdf",
                "title": "table_6",
                "document_type": "table",
                "chunk_index": 24,
                "chunk_id": "FC-007_expired",
            },
        ),
        Document(
            page_content="FC-029 | Fare: Premium | Market: TATL | Effective From: 2027-01-01 (future) | Effective To: none | Change: Free | Refund: 100% | Checked Bag: 2x32 kg.",
            metadata={
                "source": "airline_case_study.pdf",
                "title": "table_7",
                "document_type": "table",
                "chunk_index": 29,
                "chunk_id": "FC-029_future",
            },
        ),
        Document(
            page_content="FC-027 | Fare: Standard | Market: TATL | Effective From: 2026-06-01 | Effective To: none | Change: $100 | Refund: 50% fare | Checked Bag: 1x23 kg.",
            metadata={
                "source": "airline_case_study.pdf",
                "title": "table_7",
                "document_type": "table",
                "chunk_index": 27,
                "chunk_id": "FC-027_active",
            },
        ),
    ]

    # 3. Section 11.3 Regional Passenger Rights: RR-01 Northern Union vs RR-04 Gulf & Africa Compact
    q3 = "What compensation and care apply for a 4-hour arrival delay in Northern Union?"
    q3_candidates = (
        "passenger_rights",
        [
            Document(
                page_content="RR-04 | Gulf & Africa Compact (GAC) | Delay Trigger: Arrival delay >= 5 h | Compensation: $250 on request; none for weather/ATC | Care: Meals after 4 h; hotel if overnight.",
                metadata={
                    "source": "airline_case_study.pdf",
                    "title": "table_11",
                    "document_type": "table",
                    "chunk_index": 36,
                    "chunk_id": "RR-04_GAC",
                },
            ),
            Document(
                page_content="RR-01 | Northern Union (NU) | Delay Trigger: Arrival delay >= 3 h | Compensation: EUR 250 (<1,500 km), EUR 400 (1,500-3,500 km), EUR 600 (>3,500 km) | Care: Meals after 2 h; hotel if overnight.",
                metadata={
                    "source": "airline_case_study.pdf",
                    "title": "table_11",
                    "document_type": "table",
                    "chunk_index": 35,
                    "chunk_id": "RR-01_NU",
                },
            ),
        ],
    )

    # 4. Section 11.5 Map 3: Skyharbour (SKH) Two-Terminal Transfer
    q4 = "How do I transfer from Terminal 1 to Terminal 2 at Skyharbour (SKH)?"
    q4_candidates = (
        "airport_transfer",
        [
            Document(
                page_content="Meridian Hub (MRD) Terminal Transfer Map: Terminals A, B and C are linked by Skytrain (A->B 4 min, B->C 3 min, A->C 7 min via B).",
                metadata={
                    "source": "airline_case_study.pdf",
                    "section": "11.5 Three Airport Transfer Maps",
                    "chunk_index": 41,
                    "chunk_id": "map_MRD",
                },
            ),
            Document(
                page_content="Skyharbour (SKH) Two-Terminal Transfer (map_SKH_two_terminal_bus.png): 1. Exit security (leave airside) at Terminal 1, 2. Take landside bus (15 min) to Terminal 2, 3. Re-screening at Terminal 2, 4. Board connecting flight. Allow at least 90 minutes for self-transfer.",
                metadata={
                    "source": "airline_case_study.pdf",
                    "section": "11.5 Three Airport Transfer Maps",
                    "chunk_index": 43,
                    "chunk_id": "map_SKH",
                },
            ),
        ],
    )

    # 5. Section 11.5 Connection-Time Chart: Meridian Hub (MRD) Intl -> Dom
    q5 = "What is the minimum connection time at Meridian Hub (MRD) for International to Domestic transfers?"
    q5_candidates = [
        Document(
            page_content="Minimum Connection Time Chart: NVL (Novara Lakes) Dom->Dom 35 min, Dom->Intl 45 min, Intl->Dom 50 min, Intl->Intl 45 min, Self-transfer 120 min.",
            metadata={
                "source": "airline_case_study.pdf",
                "title": "table_14",
                "document_type": "table",
                "chunk_index": 45,
                "chunk_id": "MCT_NVL",
            },
        ),
        Document(
            page_content="Minimum Connection Time Chart: MRD (Meridian Hub) Dom->Dom 45 min, Dom->Intl 60 min, Intl->Dom 75 min, Intl->Intl 60 min, Self-transfer (separate tickets) 150 min.",
            metadata={
                "source": "airline_case_study.pdf",
                "title": "table_14",
                "document_type": "table",
                "chunk_index": 44,
                "chunk_id": "MCT_MRD",
            },
        ),
    ]

    return [
        (q1, "baggage_chennai_london", q1_candidates),
        (q2, "FC-027_active", q2_candidates),
        (q3, "RR-01_NU", q3_candidates),
        (q4, "map_SKH", q4_candidates),
        (q5, "MCT_MRD", q5_candidates),
    ]


def _get_first_doc_id(
    candidates: Union[Tuple[str, List[Document]], List[Document]]
) -> str:
    docs = candidates[1] if isinstance(candidates, tuple) else candidates
    if not docs:
        return "none"
    return str(docs[0].metadata.get("chunk_id", "unknown"))


def run_experiment():
    reranker = MeridianReranker(top_k=2, reference_date=date(2026, 11, 18))
    dataset = get_case_study_test_cases()

    baseline_hits = 0
    reranked_hits = 0
    total_latency = 0.0

    print("=== MERIDIAN AIR (HYBRID RETRIEVER -> FLASHRANK) RERANKING EXPERIMENT ===\n")
    for query, expected_id, candidates in dataset:
        baseline_top1 = _get_first_doc_id(candidates)
        if baseline_top1 == expected_id:
            baseline_hits += 1

        reranked_docs, latency_ms = reranker.rerank(query, candidates)
        total_latency += latency_ms
        top_doc = reranked_docs[0]
        reranked_top1 = top_doc.metadata.get("chunk_id", "unknown")
        if reranked_top1 == expected_id:
            reranked_hits += 1

        intent_tag = top_doc.metadata.get("detected_intent", "hybrid")
        print(f"Query: {query}")
        print(f"  Retriever Top-1 : {baseline_top1} (intent={intent_tag})")
        print(
            f"  Reranked Top-1  : {reranked_top1} "
            f"[{top_doc.metadata['document_id']} | "
            f"Section: {top_doc.metadata['section']} | "
            f"Status: {top_doc.metadata['status']} | "
            f"rerank={top_doc.metadata['rerank_score']}]"
        )
        print(f"  Latency         : {latency_ms} ms\n")

    n = len(dataset)
    print("=== SUMMARY FOR MEMBER 7 EVALUATION REPORT ===")
    print(f"Baseline Precision@1 (Before Reranker) : {baseline_hits}/{n} ({baseline_hits/n:.0%})")
    print(f"Reranked Precision@1 (FlashRank)       : {reranked_hits}/{n} ({reranked_hits/n:.0%})")
    print(f"Average Rerank Latency                 : {total_latency/n:.2f} ms")


if __name__ == "__main__":
    run_experiment()