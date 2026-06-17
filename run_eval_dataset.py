import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from chat_profesional import init_backend, ask_case


INPUT_CSV = "ground_truth_long.csv"
OUTPUT_CSV = "rag_eval_dataset.csv"


def serialize_contexts(result: Dict[str, Any]) -> str:
    retrieved_turns = result.get("retrieved_turns", []) or []

    contexts: List[str] = []
    for turn in retrieved_turns:
        text = (turn.get("text") or "").strip()
        if text:
            contexts.append(text)

    return json.dumps(contexts, ensure_ascii=False)


def extract_answer(result: Dict[str, Any]) -> str:
    answer = (result.get("answer") or "").strip()
    if answer:
        return answer

    answer_with_refs = (result.get("answer_with_refs") or "").strip()
    if answer_with_refs:
        return answer_with_refs

    return ""


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RAG evaluation dataset generation.")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of rows to process from ground_truth_long.csv",
    )
    args = parser.parse_args()

    input_path = Path(INPUT_CSV)
    output_path = Path(OUTPUT_CSV)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    init_backend()

    df = pd.read_csv(input_path)

    required_columns = {
        "interview_id",
        "question_id",
        "question",
        "ground_truth",
    }
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {INPUT_CSV}: {sorted(missing)}")

    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit must be greater than 0")
        df = df.head(args.limit).copy()

    output_rows: List[Dict[str, Any]] = []

    total = len(df)
    print(f"Processing {total} rows...")

    for idx, row in df.iterrows():
        raw_interview_id = row["interview_id"]
        backend_interview_id = f"chat_random_{raw_interview_id}"
        question_id = row["question_id"]
        question = row["question"]
        ground_truth = row["ground_truth"]
        notes = row["notes"] if "notes" in df.columns else ""

        print(
            f"[{idx + 1}/{total}] "
            f"interview_id={raw_interview_id} "
            f"backend_id={backend_interview_id} "
            f"question_id={question_id}"
        )

        try:
            result = ask_case(
                question=question,
                interview_id=backend_interview_id,
                history=[],
                thread_id=f"eval_{raw_interview_id}_{question_id}",
            )

            answer = extract_answer(result)
            contexts_json = serialize_contexts(result)

            output_rows.append(
                {
                    "interview_id": raw_interview_id,
                    "backend_interview_id": backend_interview_id,
                    "question_id": question_id,
                    "question": question,
                    "ground_truth": ground_truth,
                    "answer": answer,
                    "contexts": contexts_json,
                    "notes": notes,
                    "error": "",
                }
            )

        except Exception as e:
            output_rows.append(
                {
                    "interview_id": raw_interview_id,
                    "backend_interview_id": backend_interview_id,
                    "question_id": question_id,
                    "question": question,
                    "ground_truth": ground_truth,
                    "answer": "",
                    "contexts": json.dumps([], ensure_ascii=False),
                    "notes": notes,
                    "error": str(e),
                }
            )

    out_df = pd.DataFrame(output_rows)
    out_df.to_csv(output_path, index=False, encoding="utf-8")

    print(f"\nSaved evaluation dataset to: {output_path.resolve()}")
    print(f"Rows written: {len(out_df)}")


if __name__ == "__main__":
    main()