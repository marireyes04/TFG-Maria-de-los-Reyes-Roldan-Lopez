# run_ragas_eval.py

import argparse
import ast
import json
import math
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from huggingface_hub import InferenceClient
from langchain_core.embeddings import Embeddings
from langchain_openai import ChatOpenAI
from ragas import EvaluationDataset, evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    AnswerCorrectness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
    Faithfulness,
)

# =========================
# Configuración HF
# =========================

HF_TOKEN = os.getenv("HF_TOKEN", "").strip()

# Modelo juez
JUDGE_MODEL = "Qwen/Qwen2.5-7B-Instruct"

# Embeddings para answer_correctness
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Endpoint OpenAI-compatible de Hugging Face
HF_ROUTER_BASE_URL = "https://router.huggingface.co/v1"

# Métricas soportadas
SUPPORTED_METRICS = {
    "faithfulness": Faithfulness,
    "answer_relevancy": AnswerRelevancy,
    "context_precision": ContextPrecision,
    "context_recall": ContextRecall,
    "answer_correctness": AnswerCorrectness,
}


# =========================
# Helpers
# =========================

def parse_contexts(value: Any) -> List[str]:
    """
    Convierte la columna contexts a list[str].

    Casos soportados:
    - JSON string: '["ctx1", "ctx2"]'
    - Python literal string: "['ctx1', 'ctx2']"
    - lista ya parseada
    - string simple -> [string]
    - vacío / NaN -> []
    """
    if value is None:
        return []

    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]

    if pd.isna(value):
        return []

    text = str(value).strip()
    if not text:
        return []

    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return [str(x).strip() for x in parsed if str(x).strip()]
        if isinstance(parsed, str) and parsed.strip():
            return [parsed.strip()]
    except json.JSONDecodeError:
        pass

    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, list):
            return [str(x).strip() for x in parsed if str(x).strip()]
        if isinstance(parsed, str) and parsed.strip():
            return [parsed.strip()]
    except (ValueError, SyntaxError):
        pass

    return [text]


def is_nonempty_text(value: Any) -> bool:
    if value is None:
        return False
    if pd.isna(value):
        return False
    return bool(str(value).strip())


def safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        if isinstance(value, str) and not value.strip():
            return None
        x = float(value)
        if math.isnan(x):
            return None
        return x
    except (TypeError, ValueError):
        return None


class HFInferenceEmbeddings(Embeddings):
    """
    Wrapper sencillo compatible con LangChain Embeddings
    usando Hugging Face InferenceClient.feature_extraction().
    """

    def __init__(self, token: str, model: str):
        self.client = InferenceClient(api_key=token)
        self.model = model

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        vectors: List[List[float]] = []
        for text in texts:
            result = self.client.feature_extraction(text, model=self.model)
            if hasattr(result, "tolist"):
                result = result.tolist()

            if isinstance(result, list) and result and isinstance(result[0], list):
                if len(result) == 1:
                    result = result[0]

            vectors.append([float(x) for x in result])

        return vectors

    def embed_query(self, text: str) -> List[float]:
        result = self.client.feature_extraction(text, model=self.model)
        if hasattr(result, "tolist"):
            result = result.tolist()

        if isinstance(result, list) and result and isinstance(result[0], list):
            if len(result) == 1:
                result = result[0]

        return [float(x) for x in result]


def build_metrics(metric_names: List[str], llm, embeddings):
    metrics = []
    for name in metric_names:
        if name not in SUPPORTED_METRICS:
            raise ValueError(
                f"Métrica no soportada: {name}. "
                f"Opciones válidas: {list(SUPPORTED_METRICS.keys())}"
            )

        metric_cls = SUPPORTED_METRICS[name]

        if name == "answer_correctness":
            metrics.append(metric_cls(llm=llm, embeddings=embeddings))
        else:
            metrics.append(metric_cls(llm=llm))

    return metrics


def summarize_metrics(rows_df: pd.DataFrame, metric_names: List[str]) -> pd.DataFrame:
    summary_rows = []

    for metric_name in metric_names:
        if metric_name not in rows_df.columns:
            summary_rows.append(
                {
                    "metric": metric_name,
                    "mean": None,
                    "n_valid": 0,
                    "n_nan": len(rows_df),
                    "coverage": 0.0,
                }
            )
            continue

        values = rows_df[metric_name].apply(safe_float)
        n_valid = values.notna().sum()
        n_nan = values.isna().sum()
        coverage = (n_valid / len(rows_df)) if len(rows_df) else 0.0
        mean_value = values.dropna().mean() if n_valid > 0 else None

        summary_rows.append(
            {
                "metric": metric_name,
                "mean": mean_value,
                "n_valid": int(n_valid),
                "n_nan": int(n_nan),
                "coverage": coverage,
            }
        )

    return pd.DataFrame(summary_rows)


# =========================
# Main
# =========================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("rag_eval_dataset.csv"),
        help="CSV generado por run_eval_dataset.py",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("ragas_eval_outputs"),
        help="Directorio de salida",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Número máximo de filas a evaluar",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["faithfulness", "answer_relevancy"],
        help=(
            "Lista de métricas a ejecutar. "
            "Opciones: faithfulness answer_relevancy "
            "context_precision context_recall answer_correctness"
        ),
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Baraja las filas antes de aplicar el límite",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Semilla para shuffle",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Falla si alguna métrica no produce ningún valor válido",
    )
    parser.add_argument(
    "--start-row",
    type=int,
    default=0,
    help="Índice de fila desde el que empezar a seleccionar",
    )
    args = parser.parse_args()

    if not HF_TOKEN:
        raise ValueError(
            "No se encontró HF_TOKEN. Define la variable de entorno antes de ejecutar."
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # ----------
    # 1) Leer CSV original
    # ----------
    raw_df = pd.read_csv(args.input)
    n_input_rows = len(raw_df)

    # ----------
    # 2) Filtrado básico
    # ----------
    df = raw_df.copy()

    if "error" in df.columns:
        df = df[df["error"].isna() | (df["error"].astype(str).str.strip() == "")].copy()

    required_source_cols = ["question", "answer", "ground_truth", "contexts"]
    missing = [c for c in required_source_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas obligatorias en el CSV: {missing}")

    df["contexts_parsed"] = df["contexts"].apply(parse_contexts)

    df = df[
        df["question"].apply(is_nonempty_text)
        & df["answer"].apply(is_nonempty_text)
        & df["ground_truth"].apply(is_nonempty_text)
        & (df["contexts_parsed"].apply(len) > 0)
    ].copy()

    n_valid_after_cleaning = len(df)

    if df.empty:
        raise ValueError("No quedan filas válidas tras el filtrado.")

    # ----------
    # 3) Selección de subconjunto
    # ----------
    if args.shuffle:
        df = df.sample(frac=1.0, random_state=args.random_seed).reset_index(drop=True)

    if args.start_row < 0:
        raise ValueError("--start-row no puede ser negativo")

    df = df.iloc[args.start_row: args.start_row + args.limit].copy()
    n_selected_rows = len(df)

    if df.empty:
        raise ValueError(
            f"No hay filas disponibles en el rango solicitado: "
            f"start_row={args.start_row}, limit={args.limit}"
        )

    # ----------
    # 4) Adaptar a esquema Ragas
    # ----------
    eval_df = pd.DataFrame(
        {
            "interview_id": df["interview_id"] if "interview_id" in df.columns else None,
            "backend_interview_id": df["backend_interview_id"] if "backend_interview_id" in df.columns else None,
            "question_id": df["question_id"] if "question_id" in df.columns else None,
            "user_input": df["question"],
            "response": df["answer"],
            "retrieved_contexts": df["contexts_parsed"],
            "reference": df["ground_truth"],
            "notes": df["notes"] if "notes" in df.columns else None,
        }
    )

    prepared_path = args.output_dir / "ragas_input_prepared.csv"
    eval_df.to_csv(prepared_path, index=False)

    # ----------
    # 5) Crear EvaluationDataset
    # ----------
    records = eval_df.to_dict(orient="records")
    ragas_dataset = EvaluationDataset.from_list(records)

    # ----------
    # 6) Configurar juez y embeddings
    # ----------
    judge_llm = ChatOpenAI(
        model=JUDGE_MODEL,
        base_url=HF_ROUTER_BASE_URL,
        api_key=HF_TOKEN,
        temperature=0.0,
        max_completion_tokens=512,
    )
    llm = LangchainLLMWrapper(judge_llm)

    hf_embeddings = HFInferenceEmbeddings(
        token=HF_TOKEN,
        model=EMBEDDING_MODEL,
    )
    embeddings = LangchainEmbeddingsWrapper(hf_embeddings)

    metric_names = args.metrics
    metrics = build_metrics(metric_names, llm, embeddings)

    # ----------
    # 7) Evaluación
    # ----------
    result = evaluate(
        dataset=ragas_dataset,
        metrics=metrics,
        llm=llm,
        embeddings=embeddings,
        raise_exceptions=False,
        show_progress=True,
    )

    # ----------
    # 8) Resultados por fila
    # ----------
    rows_df = None

    if hasattr(result, "to_pandas"):
        rows_df = result.to_pandas()
    elif hasattr(result, "dataset") and hasattr(result.dataset, "to_pandas"):
        rows_df = result.dataset.to_pandas()

    if rows_df is None:
        rows_df = eval_df.copy()

    # Añadir metadatos útiles
    for col in [
        "interview_id",
        "backend_interview_id",
        "question_id",
        "user_input",
        "response",
        "reference",
        "notes",
    ]:
        if col not in rows_df.columns and col in eval_df.columns:
            rows_df[col] = eval_df[col].values

    # Añadir una marca útil: si la fila tiene al menos una métrica válida
    available_metric_cols = [m for m in metric_names if m in rows_df.columns]
    if available_metric_cols:
        rows_df["has_any_metric_value"] = rows_df[available_metric_cols].apply(
            lambda row: any(safe_float(v) is not None for v in row),
            axis=1,
        )
    else:
        rows_df["has_any_metric_value"] = False

    rows_path = args.output_dir / "ragas_results_rows.csv"
    rows_df.to_csv(rows_path, index=False)

    # ----------
    # 9) Resumen de métricas
    # ----------
    summary_df = summarize_metrics(rows_df, metric_names)
    summary_path = args.output_dir / "ragas_results_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    # ----------
    # 10) Metadatos de ejecución
    # ----------
    run_metadata: Dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
        "input_file": str(args.input),
        "output_dir": str(args.output_dir),
        "judge_model": JUDGE_MODEL,
        "embedding_model": EMBEDDING_MODEL if "answer_correctness" in metric_names else None,
        "metrics_requested": metric_names,
        "n_input_rows": n_input_rows,
        "n_valid_after_cleaning": n_valid_after_cleaning,
        "n_selected_rows": n_selected_rows,
        "n_rows_output": int(len(rows_df)),
        "n_rows_with_any_metric_value": int(rows_df["has_any_metric_value"].sum()),
        "start_row": args.start_row,
    }

    # Añadir cobertura por métrica al metadata
    metric_coverage = {}
    for _, row in summary_df.iterrows():
        metric_coverage[row["metric"]] = {
            "mean": row["mean"],
            "n_valid": int(row["n_valid"]),
            "n_nan": int(row["n_nan"]),
            "coverage": row["coverage"],
        }
    run_metadata["metric_coverage"] = metric_coverage

    metadata_path = args.output_dir / "ragas_run_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(run_metadata, f, ensure_ascii=False, indent=2)

    # ----------
    # 11) Validación estricta opcional
    # ----------
    if args.strict:
        metrics_without_values = [
            row["metric"] for _, row in summary_df.iterrows() if int(row["n_valid"]) == 0
        ]
        if metrics_without_values:
            raise RuntimeError(
                "Estas métricas no devolvieron ningún valor válido: "
                + ", ".join(metrics_without_values)
            )

    # ----------
    # 12) Salida por consola
    # ----------
    print("\nEvaluación completada.")
    print(f"Filas de entrada: {n_input_rows}")
    print(f"Filas válidas tras limpieza: {n_valid_after_cleaning}")
    print(f"Filas seleccionadas: {n_selected_rows}")
    print(f"Filas con al menos una métrica válida: {int(rows_df['has_any_metric_value'].sum())}")
    print(f"Dataset preparado: {prepared_path}")
    print(f"Resultados por fila: {rows_path}")
    print(f"Resumen global: {summary_path}")
    print(f"Metadatos de ejecución: {metadata_path}")

    print("\nResumen rápido por métrica:")
    for _, row in summary_df.iterrows():
        print(
            f"- {row['metric']}: mean={row['mean']} | "
            f"n_valid={int(row['n_valid'])} | "
            f"coverage={row['coverage']:.2f}"
        )


if __name__ == "__main__":
    main()