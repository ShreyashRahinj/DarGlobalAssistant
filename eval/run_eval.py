"""Evaluation script: retrieval hit rate on a held-out question set.

Run locally after building the index:
    python eval/run_eval.py

Reports:
  - Retrieval hit rate: % of questions where at least one retrieved chunk
    matches the expected source (darglobal / wasalt / both).
  - Writes eval/results.md (fill in groundedness manually or extend with an LLM judge).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.rag import Retriever  # noqa: E402

QUESTIONS = Path(__file__).resolve().parent / "questions.jsonl"
RESULTS = Path(__file__).resolve().parent / "results.md"


def load_questions() -> list[dict]:
    qs = []
    with open(QUESTIONS, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                qs.append(json.loads(line))
    return qs


def source_match(expected: str, retrieved_sources: list[str]) -> bool:
    if expected == "both":
        return "darglobal" in retrieved_sources and "wasalt" in retrieved_sources
    return expected in retrieved_sources


def main() -> None:
    if not QUESTIONS.exists():
        print(f"questions file not found: {QUESTIONS}")
        sys.exit(1)

    retriever = Retriever()
    if not retriever.load():
        print("No index found. Build it first: run scraper, then the app builds the index at startup.")
        sys.exit(1)

    questions = load_questions()
    hits = 0
    details = []

    for q in questions:
        results = retriever.retrieve(q["question"], source_filter="all")
        retrieved_sources = [c.source for c, _ in results]
        hit = source_match(q.get("expected_source", "all"), retrieved_sources)
        hits += 1 if hit else 0
        details.append(
            {
                "question": q["question"],
                "expected": q.get("expected_source"),
                "hit": hit,
                "retrieved_sources": retrieved_sources,
                "n_retrieved": len(results),
            }
        )

    hit_rate = hits / len(questions) if questions else 0.0

    print(f"\nRetrieval hit rate: {hits}/{len(questions)} = {hit_rate:.1%}\n")
    for d in details:
        mark = "OK " if d["hit"] else "MISS"
        print(f"[{mark}] {d['question'][:70]}...  expected={d['expected']} got={d['retrieved_sources']}")

    report = f"""# Eval Results

Generated: retrieval hit-rate benchmark.

- Questions: {len(questions)}
- Hits: {hits}
- Hit rate: {hit_rate:.1%}

## Per-question detail

"""
    for d in details:
        mark = "OK" if d["hit"] else "MISS"
        report += f"- [{mark}] {d['question']} (expected: {d['expected']}, got: {d['retrieved_sources']})\n"

    report += """
## Notes

- Hit rate measures whether the expected source appeared among the top retrieved chunks.
- Groundedness (whether answers are supported by context) should be checked manually
  against the live /chat endpoint; an LLM-as-judge can be added here later.
"""
    RESULTS.write_text(report, encoding="utf-8")
    print(f"\nReport written to {RESULTS}")


if __name__ == "__main__":
    main()
