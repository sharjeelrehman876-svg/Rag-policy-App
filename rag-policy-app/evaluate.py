"""
Evaluation Script for TechCorp Policy RAG Application
------------------------------------------------------
Measures:
  - Groundedness: Does answer contain only info from source docs?
  - Citation Accuracy: Do citations match the answer topic?
  - Latency: p50 and p95 response times

Usage:
  1. Make sure app is running: python app.py
  2. Run: python evaluate.py
  3. Results saved to evaluation_results.json
"""

import time
import json
import statistics
import requests

BASE_URL = "http://localhost:5000"

# ── Evaluation Set: 20 questions with expected sources ─────────────────────────
EVAL_QUESTIONS = [
    {
        "id": "Q01",
        "question": "How many PTO days does a new employee get in their first year?",
        "expected_doc_ids": ["HR-001"],
        "gold_answer_keywords": ["15", "days"],
        "topic": "PTO"
    },
    {
        "id": "Q02",
        "question": "Can employees carry over unused PTO to the next year?",
        "expected_doc_ids": ["HR-001"],
        "gold_answer_keywords": ["10", "carry", "carryover"],
        "topic": "PTO"
    },
    {
        "id": "Q03",
        "question": "What happens to PTO if an employee leaves the company?",
        "expected_doc_ids": ["HR-001"],
        "gold_answer_keywords": ["payout", "final", "paycheck"],
        "topic": "PTO"
    },
    {
        "id": "Q04",
        "question": "How many days in advance must I request PTO for a week-long vacation?",
        "expected_doc_ids": ["HR-001"],
        "gold_answer_keywords": ["5", "business days"],
        "topic": "PTO"
    },
    {
        "id": "Q05",
        "question": "What are the core working hours for remote employees?",
        "expected_doc_ids": ["HR-002"],
        "gold_answer_keywords": ["10:00", "3:00", "core hours"],
        "topic": "Remote Work"
    },
    {
        "id": "Q06",
        "question": "How many days per week must hybrid employees come into the office?",
        "expected_doc_ids": ["HR-002"],
        "gold_answer_keywords": ["3", "days", "week"],
        "topic": "Remote Work"
    },
    {
        "id": "Q07",
        "question": "Does the company provide equipment for remote workers?",
        "expected_doc_ids": ["HR-002"],
        "gold_answer_keywords": ["laptop", "headset", "monitor"],
        "topic": "Remote Work"
    },
    {
        "id": "Q08",
        "question": "How much internet reimbursement do remote employees get?",
        "expected_doc_ids": ["HR-002"],
        "gold_answer_keywords": ["50", "month"],
        "topic": "Remote Work"
    },
    {
        "id": "Q09",
        "question": "What is the meal allowance for dinner during business travel?",
        "expected_doc_ids": ["FIN-001"],
        "gold_answer_keywords": ["50", "dinner"],
        "topic": "Expenses"
    },
    {
        "id": "Q10",
        "question": "Do I need a receipt for a $20 purchase?",
        "expected_doc_ids": ["FIN-001"],
        "gold_answer_keywords": ["25", "receipt"],
        "topic": "Expenses"
    },
    {
        "id": "Q11",
        "question": "How many company holidays does TechCorp observe?",
        "expected_doc_ids": ["HR-005"],
        "gold_answer_keywords": ["11", "holidays"],
        "topic": "Holidays"
    },
    {
        "id": "Q12",
        "question": "Does TechCorp give floating holidays?",
        "expected_doc_ids": ["HR-005"],
        "gold_answer_keywords": ["2", "floating"],
        "topic": "Holidays"
    },
    {
        "id": "Q13",
        "question": "How many bereavement days do I get for the death of a parent?",
        "expected_doc_ids": ["HR-005"],
        "gold_answer_keywords": ["5", "days", "parent"],
        "topic": "Leave"
    },
    {
        "id": "Q14",
        "question": "What are the password requirements for company systems?",
        "expected_doc_ids": ["IT-001"],
        "gold_answer_keywords": ["12", "characters", "password"],
        "topic": "Security"
    },
    {
        "id": "Q15",
        "question": "How often must I change my password?",
        "expected_doc_ids": ["IT-001"],
        "gold_answer_keywords": ["90", "days"],
        "topic": "Security"
    },
    {
        "id": "Q16",
        "question": "What is the 401k employer match at TechCorp?",
        "expected_doc_ids": ["HR-007"],
        "gold_answer_keywords": ["4%", "100%", "match"],
        "topic": "Benefits"
    },
    {
        "id": "Q17",
        "question": "How long is the primary caregiver parental leave?",
        "expected_doc_ids": ["HR-007"],
        "gold_answer_keywords": ["16", "weeks"],
        "topic": "Benefits"
    },
    {
        "id": "Q18",
        "question": "When is the annual performance review conducted?",
        "expected_doc_ids": ["HR-004"],
        "gold_answer_keywords": ["December", "annual"],
        "topic": "Performance"
    },
    {
        "id": "Q19",
        "question": "What salary increase does an employee with rating 5 get?",
        "expected_doc_ids": ["HR-004"],
        "gold_answer_keywords": ["6", "8", "%"],
        "topic": "Performance"
    },
    {
        "id": "Q20",
        "question": "What is TechCorp's policy on using public WiFi?",
        "expected_doc_ids": ["IT-001", "HR-002"],
        "gold_answer_keywords": ["VPN", "public", "WiFi"],
        "topic": "Security"
    },
]


# ── Evaluation Functions ────────────────────────────────────────────────────────

def check_groundedness(answer: str, question_data: dict) -> bool:
    """
    Simple keyword-based groundedness check:
    Checks if the answer contains expected keywords from the gold answer.
    In production, this would use an LLM judge.
    """
    answer_lower = answer.lower()
    keywords = question_data.get("gold_answer_keywords", [])
    if not keywords:
        return True
    matches = sum(1 for kw in keywords if kw.lower() in answer_lower)
    return matches >= len(keywords) * 0.5  # at least 50% keyword match


def check_citation_accuracy(citations: list, expected_doc_ids: list) -> bool:
    """Check if at least one expected doc ID appears in citations."""
    if not expected_doc_ids:
        return True
    cited_ids = {c["doc_id"] for c in citations}
    return bool(cited_ids.intersection(set(expected_doc_ids)))


def check_out_of_scope_refusal(answer: str) -> bool:
    """Check if out-of-scope questions are properly refused."""
    refusal_phrases = [
        "i can only answer",
        "not in our policy",
        "cannot find",
        "not covered",
    ]
    answer_lower = answer.lower()
    return any(p in answer_lower for p in refusal_phrases)


# ── Main Evaluation Loop ────────────────────────────────────────────────────────

def run_evaluation():
    print("=" * 60)
    print("TechCorp RAG Evaluation")
    print("=" * 60)
    print(f"Evaluating {len(EVAL_QUESTIONS)} questions...\n")

    results = []
    latencies = []
    groundedness_scores = []
    citation_scores = []

    for q_data in EVAL_QUESTIONS:
        q_id = q_data["id"]
        question = q_data["question"]

        try:
            start = time.time()
            resp = requests.post(
                f"{BASE_URL}/chat",
                json={"question": question},
                timeout=30,
            )
            elapsed_ms = round((time.time() - start) * 1000, 2)

            if resp.status_code != 200:
                print(f"  [{q_id}] ❌ HTTP {resp.status_code}")
                results.append({"id": q_id, "error": f"HTTP {resp.status_code}"})
                continue

            data = resp.json()
            answer = data.get("answer", "")
            citations = data.get("citations", [])

            # Score
            grounded = check_groundedness(answer, q_data)
            cit_accurate = check_citation_accuracy(citations, q_data["expected_doc_ids"])

            latencies.append(elapsed_ms)
            groundedness_scores.append(grounded)
            citation_scores.append(cit_accurate)

            status = "✅" if grounded and cit_accurate else "⚠️"
            print(f"  [{q_id}] {status} {question[:60]}...")
            print(f"         Latency: {elapsed_ms}ms | Grounded: {grounded} | Citation OK: {cit_accurate}")
            print(f"         Answer: {answer[:120]}...")
            print()

            results.append({
                "id": q_id,
                "topic": q_data["topic"],
                "question": question,
                "answer": answer,
                "citations": citations,
                "latency_ms": elapsed_ms,
                "groundedness": grounded,
                "citation_accuracy": cit_accurate,
            })

        except Exception as e:
            print(f"  [{q_id}] ❌ Error: {e}")
            results.append({"id": q_id, "error": str(e)})

    # ── Summary ────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)

    n = len(latencies)
    if n > 0:
        groundedness_pct = round(sum(groundedness_scores) / n * 100, 1)
        citation_pct = round(sum(citation_scores) / n * 100, 1)
        p50 = round(statistics.median(latencies), 2)
        p95 = round(sorted(latencies)[int(len(latencies) * 0.95)], 2) if len(latencies) >= 20 else round(max(latencies), 2)

        print(f"Questions evaluated:    {n}/{len(EVAL_QUESTIONS)}")
        print(f"Groundedness:           {groundedness_pct}%")
        print(f"Citation Accuracy:      {citation_pct}%")
        print(f"Latency p50:            {p50} ms")
        print(f"Latency p95:            {p95} ms")
        print(f"Avg Latency:            {round(statistics.mean(latencies), 2)} ms")

        summary = {
            "total_questions": n,
            "groundedness_pct": groundedness_pct,
            "citation_accuracy_pct": citation_pct,
            "latency_p50_ms": p50,
            "latency_p95_ms": p95,
            "latency_avg_ms": round(statistics.mean(latencies), 2),
        }
    else:
        summary = {"error": "No successful evaluations"}

    # Save results
    output = {"summary": summary, "results": results}
    with open("evaluation_results.json", "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nFull results saved to: evaluation_results.json")
    print("=" * 60)


if __name__ == "__main__":
    run_evaluation()
