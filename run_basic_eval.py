# run_basic_eval.py
"""
Basic evaluation for the CDSS RAG system.

This script provides a robust evaluation layer that does not depend on Ragas.
It can either:
1) generate answers by calling chat_profesional.ask_case(), or
2) score an existing rag_eval_dataset.csv without calling the LLM.

Outputs:
- basic_eval_outputs/basic_eval_rows.csv
- basic_eval_outputs/basic_eval_summary.csv
- basic_eval_outputs/basic_eval_metadata.json

Example usage:
    python run_basic_eval.py --input ground_truth_long.csv --limit 40

Score an already generated CSV:
    python run_basic_eval.py --input rag_eval_dataset.csv --from-existing --output-dir basic_eval_outputs_existing
"""

import argparse
import json
import math
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import pandas as pd

from chat_profesional import init_backend, ask_case


RISK_KEYWORDS = {
    "suicidal",
    "suicide",
    "self-harm",
    "self harm",
    "self-injury",
    "self injury",
    "risk",
    "danger",
    "safety",
    "urgent",
    "crisis",
    "emergency",
    "harm himself",
    "harm herself",
    "harm themselves",
}

ABSENCE_MARKERS = {
    "not mentioned",
    "not explicitly mentioned",
    "no mention",
    "no explicit mention",
    "not stated",
    "no evidence",
    "not present",
    "not discussed",
    "not reported",
    "not indicated",
}


def normalize_text(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


def tokenize(text: Any) -> Set[str]:
    text = normalize_text(text).lower()
    tokens = re.findall(r"[a-zA-Z][a-zA-Z\-']{2,}", text)
    stopwords = {
        "the", "and", "that", "this", "with", "from", "are", "was", "were",
        "for", "not", "but", "his", "her", "their", "patient", "interview",
        "mentioned", "explicitly", "there", "about", "into", "only", "does",
        "did", "has", "have", "had", "been", "being", "what", "when", "where",
        "which", "who", "whom", "why", "how", "its", "it's", "they", "them",
        "she", "him", "you", "your", "our", "can", "could", "should", "would",
    }
    return {tok for tok in tokens if tok not in stopwords}


def lexical_overlap(answer: Any, ground_truth: Any) -> float:
    answer_tokens = tokenize(answer)
    truth_tokens = tokenize(ground_truth)

    if not truth_tokens:
        return 0.0

    return round(len(answer_tokens & truth_tokens) / len(truth_tokens), 4)


def parse_contexts(value: Any) -> List[str]:
    if value is None:
        return []

    if isinstance(value, list):
        return [normalize_text(x) for x in value if normalize_text(x)]

    if isinstance(value, float) and math.isnan(value):
        return []

    text = normalize_text(value)
    if not text:
        return []

    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return [normalize_text(x) for x in parsed if normalize_text(x)]
        if isinstance(parsed, str) and normalize_text(parsed):
            return [normalize_text(parsed)]
    except Exception:
        pass

    return [text]


def serialize_contexts(result: Dict[str, Any]) -> str:
    contexts: List[str] = []

    for turn in result.get("retrieved_turns", []) or []:
        text = normalize_text(turn.get("text", ""))
        if text:
            contexts.append(text)

    return json.dumps(contexts, ensure_ascii=False)


def is_nonempty(value: Any) -> bool:
    return bool(normalize_text(value))


def is_risk_question(question: Any, question_id: Any = "") -> bool:
    q = normalize_text(question).lower()
    qid = normalize_text(question_id).upper()

    if qid in {"Q2", "Q5"}:
        return True

    return any(keyword in q for keyword in RISK_KEYWORDS)


def contains_absence_marker(text: Any) -> bool:
    value = normalize_text(text).lower()
    return any(marker in value for marker in ABSENCE_MARKERS)


def expected_absence(ground_truth: Any) -> bool:
    value = normalize_text(ground_truth).lower()

    # Common ground-truth wording: "No, suicidal thoughts are not mentioned."
    if value.startswith("no,"):
        return True

    return contains_absence_marker(value)


def risk_alignment(answer: Any, ground_truth: Any, question: Any, question_id: Any = "") -> Optional[bool]:
    if not is_risk_question(question, question_id):
        return None

    gt_absent = expected_absence(ground_truth)
    ans_absent = contains_absence_marker(answer) or normalize_text(answer).lower().startswith("no,")

    if gt_absent:
        return bool(ans_absent)

    # Positive / present risk case: answer should not deny or mark absence.
    return not ans_absent and is_nonempty(answer)


def extract_answer(result: Dict[str, Any]) -> str:
    answer = normalize_text(result.get("answer", ""))
    if answer:
        return answer

    return normalize_text(result.get("answer_with_refs", ""))


def build_backend_interview_id(raw_interview_id: Any, prefix: str) -> str:
    raw = normalize_text(raw_interview_id)
    if raw.startswith(prefix):
        return raw
    return f"{prefix}{raw}"


def evaluate_generated_row(
    row: pd.Series,
    backend_prefix: str,
) -> Dict[str, Any]:
    raw_interview_id = row["interview_id"]
    backend_interview_id = (
        row["backend_interview_id"]
        if "backend_interview_id" in row and is_nonempty(row.get("backend_interview_id"))
        else build_backend_interview_id(raw_interview_id, backend_prefix)
    )

    question_id = row.get("question_id", "")
    question = row["question"]
    ground_truth = row["ground_truth"]
    notes = row.get("notes", "")

    started = time.time()

    try:
        result = ask_case(
            question=question,
            interview_id=backend_interview_id,
            history=[],
            thread_id=f"basic_eval_{raw_interview_id}_{question_id}",
        )

        answer = extract_answer(result)
        contexts_json = serialize_contexts(result)
        contexts = parse_contexts(contexts_json)
        evidence_items = result.get("evidence", []) or []
        retrieved_turns = result.get("retrieved_turns", []) or []

        error = ""
        elapsed = result.get("elapsed_sec")
        if elapsed is None:
            elapsed = round(time.time() - started, 3)

        manual_rule = normalize_text(result.get("manual_rule", ""))

        output = {
            "interview_id": raw_interview_id,
            "backend_interview_id": backend_interview_id,
            "question_id": question_id,
            "question": question,
            "ground_truth": ground_truth,
            "answer": answer,
            "contexts": contexts_json,
            "notes": notes,
            "error": error,
            "answer_generated": is_nonempty(answer),
            "has_contexts": len(contexts) > 0,
            "num_contexts": len(contexts),
            "num_retrieved_turns": len(retrieved_turns),
            "has_evidence_items": len(evidence_items) > 0,
            "num_evidence_items": len(evidence_items),
            "has_manual_rule": bool(manual_rule and manual_rule.lower() != "not applicable."),
            "manual_rule": manual_rule,
            "format_ok": bool(result.get("format_ok", False)),
            "latency_sec": elapsed,
            "ground_truth_overlap": lexical_overlap(answer, ground_truth),
            "is_risk_question": is_risk_question(question, question_id),
            "risk_alignment": risk_alignment(answer, ground_truth, question, question_id),
        }

        return output

    except Exception as exc:
        answer = ""
        return {
            "interview_id": raw_interview_id,
            "backend_interview_id": backend_interview_id,
            "question_id": question_id,
            "question": question,
            "ground_truth": ground_truth,
            "answer": answer,
            "contexts": json.dumps([], ensure_ascii=False),
            "notes": notes,
            "error": str(exc),
            "answer_generated": False,
            "has_contexts": False,
            "num_contexts": 0,
            "num_retrieved_turns": 0,
            "has_evidence_items": False,
            "num_evidence_items": 0,
            "has_manual_rule": False,
            "manual_rule": "",
            "format_ok": False,
            "latency_sec": round(time.time() - started, 3),
            "ground_truth_overlap": 0.0,
            "is_risk_question": is_risk_question(question, question_id),
            "risk_alignment": None,
        }


def evaluate_existing_row(row: pd.Series) -> Dict[str, Any]:
    question = row.get("question", "")
    question_id = row.get("question_id", "")
    ground_truth = row.get("ground_truth", "")
    answer = row.get("answer", "")
    contexts = parse_contexts(row.get("contexts", ""))
    error = normalize_text(row.get("error", ""))

    return {
        "interview_id": row.get("interview_id", ""),
        "backend_interview_id": row.get("backend_interview_id", ""),
        "question_id": question_id,
        "question": question,
        "ground_truth": ground_truth,
        "answer": answer,
        "contexts": json.dumps(contexts, ensure_ascii=False),
        "notes": row.get("notes", ""),
        "error": error,
        "answer_generated": is_nonempty(answer),
        "has_contexts": len(contexts) > 0,
        "num_contexts": len(contexts),
        "num_retrieved_turns": len(contexts),
        "has_evidence_items": False,
        "num_evidence_items": 0,
        "has_manual_rule": False,
        "manual_rule": "",
        "format_ok": False,
        "latency_sec": None,
        "ground_truth_overlap": lexical_overlap(answer, ground_truth),
        "is_risk_question": is_risk_question(question, question_id),
        "risk_alignment": risk_alignment(answer, ground_truth, question, question_id),
    }


def summarize(rows_df: pd.DataFrame) -> pd.DataFrame:
    total = len(rows_df)
    if total == 0:
        return pd.DataFrame()

    def pct(series: pd.Series) -> float:
        return round(float(series.mean() * 100), 2) if len(series) else 0.0

    successful = rows_df["error"].fillna("").astype(str).str.strip().eq("")
    risk_rows = rows_df[rows_df["is_risk_question"].fillna(False).astype(bool)]
    risk_valid = risk_rows["risk_alignment"].dropna() if "risk_alignment" in risk_rows else pd.Series(dtype=bool)

    summary = [
        {"metric": "total_examples", "value": total, "description": "Rows selected for evaluation."},
        {"metric": "successful_calls", "value": int(successful.sum()), "description": "Rows without runtime error."},
        {"metric": "error_rate_pct", "value": round(float((~successful).mean() * 100), 2), "description": "Percentage of rows with runtime error."},
        {"metric": "answer_generation_rate_pct", "value": pct(rows_df["answer_generated"].fillna(False).astype(bool)), "description": "Percentage of rows with a non-empty generated answer."},
        {"metric": "context_retrieval_rate_pct", "value": pct(rows_df["has_contexts"].fillna(False).astype(bool)), "description": "Percentage of rows with at least one retrieved context."},
        {"metric": "evidence_item_rate_pct", "value": pct(rows_df["has_evidence_items"].fillna(False).astype(bool)), "description": "Percentage of rows with traceable evidence items."},
        {"metric": "format_ok_rate_pct", "value": pct(rows_df["format_ok"].fillna(False).astype(bool)), "description": "Percentage of backend answers following the required output format."},
        {"metric": "avg_retrieved_turns", "value": round(float(rows_df["num_retrieved_turns"].fillna(0).mean()), 3), "description": "Average number of retrieved turns/contexts."},
        {"metric": "avg_ground_truth_overlap", "value": round(float(rows_df["ground_truth_overlap"].fillna(0).mean()), 4), "description": "Average token overlap between generated answer and reference answer."},
        {"metric": "avg_latency_sec", "value": round(float(rows_df["latency_sec"].dropna().mean()), 3) if rows_df["latency_sec"].notna().any() else None, "description": "Average backend latency in seconds."},
        {"metric": "risk_examples", "value": int(len(risk_rows)), "description": "Rows classified as risk/suicide/self-harm-related."},
        {"metric": "risk_alignment_rate_pct", "value": round(float(risk_valid.mean() * 100), 2) if len(risk_valid) else None, "description": "Percentage of risk rows where answer polarity aligns with ground truth."},
    ]

    return pd.DataFrame(summary)


def select_subset(df: pd.DataFrame, limit: Optional[int], start_row: int, sample_per_question: Optional[int]) -> pd.DataFrame:
    if start_row < 0:
        raise ValueError("--start-row cannot be negative")

    working = df.copy()

    if sample_per_question is not None:
        if sample_per_question <= 0:
            raise ValueError("--sample-per-question must be greater than 0")
        if "question_id" not in working.columns:
            raise ValueError("--sample-per-question requires a question_id column")
        working = (
            working.groupby("question_id", group_keys=False)
            .head(sample_per_question)
            .reset_index(drop=True)
        )

    if start_row:
        working = working.iloc[start_row:].copy()

    if limit is not None:
        if limit <= 0:
            raise ValueError("--limit must be greater than 0")
        working = working.head(limit).copy()

    return working.reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run basic CDSS RAG evaluation.")
    parser.add_argument("--input", type=Path, default=Path("ground_truth_long.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("basic_eval_outputs"))
    parser.add_argument("--limit", type=int, default=40)
    parser.add_argument("--start-row", type=int, default=0)
    parser.add_argument("--sample-per-question", type=int, default=None)
    parser.add_argument("--backend-prefix", type=str, default="chat_random_")
    parser.add_argument("--from-existing", action="store_true", help="Score an existing rag_eval_dataset.csv without calling ask_case().")
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--random-seed", type=int, default=42)
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    raw_df = pd.read_csv(args.input)

    required = {"interview_id", "question_id", "question", "ground_truth"}
    if args.from_existing:
        required = required | {"answer", "contexts"}

    missing = required - set(raw_df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {args.input}: {sorted(missing)}")

    if args.shuffle:
        raw_df = raw_df.sample(frac=1.0, random_state=args.random_seed).reset_index(drop=True)

    selected_df = select_subset(
        raw_df,
        limit=args.limit,
        start_row=args.start_row,
        sample_per_question=args.sample_per_question,
    )

    if selected_df.empty:
        raise ValueError("No rows selected for evaluation.")

    if not args.from_existing:
        init_backend()

    rows: List[Dict[str, Any]] = []
    total = len(selected_df)

    print(f"Evaluating {total} rows...")
    for i, (_, row) in enumerate(selected_df.iterrows(), start=1):
        print(f"[{i}/{total}] interview_id={row.get('interview_id')} question_id={row.get('question_id')}")
        if args.from_existing:
            rows.append(evaluate_existing_row(row))
        else:
            rows.append(evaluate_generated_row(row, backend_prefix=args.backend_prefix))

    rows_df = pd.DataFrame(rows)
    summary_df = summarize(rows_df)

    rows_path = args.output_dir / "basic_eval_rows.csv"
    summary_path = args.output_dir / "basic_eval_summary.csv"
    metadata_path = args.output_dir / "basic_eval_metadata.json"

    rows_df.to_csv(rows_path, index=False, encoding="utf-8")
    summary_df.to_csv(summary_path, index=False, encoding="utf-8")

    metadata = {
        "timestamp": datetime.now().isoformat(),
        "input": str(args.input),
        "output_dir": str(args.output_dir),
        "from_existing": args.from_existing,
        "limit": args.limit,
        "start_row": args.start_row,
        "sample_per_question": args.sample_per_question,
        "n_input_rows": int(len(raw_df)),
        "n_selected_rows": int(len(selected_df)),
        "n_output_rows": int(len(rows_df)),
    }

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print("\nBasic evaluation completed.")
    print(f"Rows: {rows_path.resolve()}")
    print(f"Summary: {summary_path.resolve()}")
    print(f"Metadata: {metadata_path.resolve()}")
    print("\nSummary:")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
