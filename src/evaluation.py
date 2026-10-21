"""Calculate retrieval metrics and human-reviewed answer metrics."""
from __future__ import annotations
from pathlib import Path
import pandas as pd
from .preprocessing import parse_pages


def _bool_label(value):
    text = str(value).strip().lower()
    if text in {"yes", "y", "true", "1", "correct", "hallucination"}:
        return True
    if text in {"no", "n", "false", "0", "incorrect", "none"}:
        return False
    return None


def _retrieved_pages(value) -> list[int]:
    if value is None or str(value).strip() == "":
        return []
    return [int(x) for x in str(value).replace(",", ";").split(";") if x.strip().isdigit()]


def metrics_for_method(group: pd.DataFrame, top_k: int = 3) -> dict:
    answerable = group[group["answerable"].astype(str).str.lower().isin({"true", "yes", "1"})]
    recalls_1, recalls_k, reciprocal_ranks, retrieval_success = [], [], [], []
    for _, row in answerable.iterrows():
        relevant = set(parse_pages(row.get("expected_pages", "")))
        ranked = _retrieved_pages(row.get("retrieved_pages", ""))
        if not relevant:
            continue
        recalls_1.append(len(set(ranked[:1]) & relevant) / len(relevant))
        recalls_k.append(len(set(ranked[:top_k]) & relevant) / len(relevant))
        first_rank = next((i + 1 for i, page in enumerate(ranked[:top_k]) if page in relevant), None)
        reciprocal_ranks.append(1 / first_rank if first_rank else 0.0)
        retrieval_success.append(bool(set(ranked[:top_k]) & relevant))

    def average(values):
        return round(sum(values) / len(values), 4) if values else "NA"

    answer_labels = [_bool_label(x) for x in group.get("answer_correct", pd.Series(dtype=object))]
    reviewed_answers = [x for x in answer_labels if x is not None]
    accuracy = round(sum(reviewed_answers) / len(reviewed_answers), 4) if reviewed_answers else "NA"

    hall_labels = [_bool_label(x) for x in group.get("hallucination", pd.Series(dtype=object))]
    reviewed_hallucinations = [x for x in hall_labels if x is not None]
    hall_rate = round(sum(reviewed_hallucinations) / len(reviewed_hallucinations), 4) if reviewed_hallucinations else "NA"

    gen_reviewed = []
    for _, row in group.iterrows():
        label = _bool_label(row.get("answer_correct", ""))
        success = _bool_label(row.get("retrieval_success_at_k", ""))
        if label is not None and success is True:
            gen_reviewed.append(label)
    gen_failure = round(sum(not x for x in gen_reviewed) / len(gen_reviewed), 4) if gen_reviewed else "NA"

    retrieval_failure_rate = round(1 - sum(retrieval_success) / len(retrieval_success), 4) if retrieval_success else "NA"
    result = {
        "Recall@1": average(recalls_1),
        f"Recall@{top_k}": average(recalls_k),
        f"MRR@{top_k}": average(reciprocal_ranks),
        "Answer Accuracy": accuracy,
        "Hallucination Rate": hall_rate,
        "Retrieval-Failure Rate": retrieval_failure_rate,
        "Generation-Failure Rate": gen_failure,
        "Questions evaluated for retrieval": len(retrieval_success),
        "Answers manually reviewed": len(reviewed_answers),
    }
    return result


def build_comparison_table(results_path: str | Path = "results/results.csv", output_path: str | Path = "results/comparison_table.csv", top_k: int = 3):
    results_path = Path(results_path)
    if not results_path.exists() or results_path.stat().st_size == 0:
        table = pd.DataFrame(columns=["method", "Recall@1", f"Recall@{top_k}", f"MRR@{top_k}", "Answer Accuracy", "Hallucination Rate", "Retrieval-Failure Rate", "Generation-Failure Rate", "Questions evaluated for retrieval", "Answers manually reviewed"])
        table.to_csv(output_path, index=False)
        return table
    frame = pd.read_csv(results_path, keep_default_na=False)
    if frame.empty or "method" not in frame.columns:
        table = pd.DataFrame(columns=["method", "Recall@1", f"Recall@{top_k}", f"MRR@{top_k}", "Answer Accuracy", "Hallucination Rate", "Retrieval-Failure Rate", "Generation-Failure Rate", "Questions evaluated for retrieval", "Answers manually reviewed"])
    else:
        rows = []
        for method, group in frame.groupby("method", sort=False):
            rows.append({"method": method, **metrics_for_method(group, top_k)})
        table = pd.DataFrame(rows)
    table.to_csv(output_path, index=False)
    return table
