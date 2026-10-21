"""Run both retrievers on the benchmark and calculate comparison metrics."""
import argparse
import json
import os
from pathlib import Path
import pandas as pd
from src.preprocessing import load_documents, load_questions, parse_pages
from src.model import TfidfRetriever, SemanticRetriever, generate_answer, DEFAULT_LLM_MODEL
from src.evaluation import build_comparison_table


def main():
    parser = argparse.ArgumentParser(description="Compare TF-IDF and semantic retrieval on BIT rules.")
    parser.add_argument("--top-k", type=int, default=3, help="Number of pages to retrieve (default: 3).")
    parser.add_argument("--limit", type=int, default=0, help="Optional number of benchmark questions for a quick test.")
    parser.add_argument("--methods", nargs="+", choices=["tfidf", "semantic"], default=["tfidf", "semantic"], help="Retriever methods to run.")
    parser.add_argument("--skip-generation", action="store_true", help="Evaluate retrieval only; do not call the LLM.")
    parser.add_argument("--evaluate-only", action="store_true", help="Recalculate metrics from existing results.csv after manual review.")
    parser.add_argument("--data-dir", default="data", help="Data directory.")
    parser.add_argument("--results-dir", default="results", help="Output directory.")
    args = parser.parse_args()
    if args.top_k < 1:
        parser.error("--top-k must be at least 1")

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    results_path = results_dir / "results.csv"
    comparison_path = results_dir / "comparison_table.csv"

    if args.evaluate_only:
        table = build_comparison_table(results_path, comparison_path, args.top_k)
        print(f"Updated {comparison_path}")
        print(table.to_string(index=False) if not table.empty else "No result rows found yet.")
        return

    data_dir = Path(args.data_dir)
    documents = load_documents(data_dir / "bit_rules_extracted.txt")
    questions = load_questions(data_dir / "sample_data.csv")
    if args.limit > 0:
        questions = questions.head(args.limit)

    print(f"Loaded {len(documents)} non-empty PDF pages and {len(questions)} questions.")
    print(f"LLM model (shared by all methods): {DEFAULT_LLM_MODEL}")
    retrievers = {}
    for method in args.methods:
        print(f"Building {method} retriever...")
        retrievers[method] = TfidfRetriever(documents) if method == "tfidf" else SemanticRetriever(documents)

    rows = []
    for q_index, (_, question_row) in enumerate(questions.iterrows(), start=1):
        qid = question_row["query_id"]
        question = question_row["question"]
        expected_pages = parse_pages(question_row["expected_pages"])
        is_answerable = bool(question_row["answerable"])
        for method, retriever in retrievers.items():
            retrieved = retriever.search(question, top_k=args.top_k)
            pages = [item["page"] for item in retrieved]
            is_success = bool(set(pages) & set(expected_pages)) if is_answerable else None
            if args.skip_generation:
                answer = ""
            else:
                answer = generate_answer(question, retrieved, DEFAULT_LLM_MODEL)
            rows.append({
                "query_id": qid,
                "method": method,
                "question": question,
                "answerable": str(is_answerable).lower(),
                "expected_answer": question_row["expected_answer"],
                "expected_pages": ";".join(map(str, expected_pages)),
                "retrieved_pages": ";".join(map(str, pages)),
                "retrieved_scores": ";".join(f"{item['score']:.4f}" for item in retrieved),
                "retrieval_success_at_k": "" if is_success is None else ("yes" if is_success else "no"),
                "answer": answer,
                "llm_model": DEFAULT_LLM_MODEL if not args.skip_generation else "not_run",
                "answer_correct": "",
                "hallucination": "",
                "review_notes": "",
                "top_k": args.top_k,
            })
        print(f"[{q_index}/{len(questions)}] completed {qid}: {question}")

    pd.DataFrame(rows).to_csv(results_path, index=False, encoding="utf-8-sig")
    table = build_comparison_table(results_path, comparison_path, args.top_k)
    print(f"\nSaved row-level results: {results_path}")
    print(f"Saved comparison metrics: {comparison_path}")
    if not table.empty:
        print(table.to_string(index=False))
    if not args.skip_generation:
        print("\nNext: review each answer and fill answer_correct/hallucination with yes/no, then run:")
        print(f"python run_experiment.py --evaluate-only --top-k {args.top_k}")

if __name__ == "__main__":
    main()
