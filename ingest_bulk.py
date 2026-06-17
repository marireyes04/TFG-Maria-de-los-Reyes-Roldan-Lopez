import os
import re
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

os.environ["ANONYMIZED_TELEMETRY"] = "False"
warnings.filterwarnings("ignore")

from unstructured.partition.auto import partition
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# =========================
# CONFIG
# =========================
CASES_FOLDER = Path("data/entradas")
MANUALS_FOLDER = Path("data/manuales")
CHROMA_PATH = "data/chroma"
COLLECTION_NAME = "interviews"

EMB = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")

# Chunking (CPU-friendly)
splitter = RecursiveCharacterTextSplitter(
    chunk_size=450,
    chunk_overlap=80
)

# =========================
# SPEAKER / TURN PARSING
# =========================
SPEAKER_PATTERNS: List[Tuple[str, str]] = [
    # Spanish
    (r"^\s*paciente\s*:\s*", "patient"),
    (r"^\s*entrevistador\s*:\s*", "therapist"),
    (r"^\s*terapeuta\s*:\s*", "therapist"),

    # English
    (r"^\s*patient\s*:\s*", "patient"),
    (r"^\s*client\s*:\s*", "patient"),
    (r"^\s*interviewer\s*:\s*", "therapist"),
    (r"^\s*therapist\s*:\s*", "therapist"),
    (r"^\s*counsel(or|er)\s*:\s*", "therapist"),
    (r"^\s*respond(er|ent)\s*:\s*", "therapist"),
]

HEADER_HINTS = [
    # English
    "context:", "language:", "case_id:",
    # Spanish kept for compatibility
    "contexto:", "idioma:", "id_caso:", "id_case:"
]


def split_into_turns_with_lines(raw_text: str) -> List[Dict]:
    """
    Split raw text into speaker turns and preserve line ranges.

    Returns a list like:
    [
        {
            "speaker": "patient",
            "text": "...",
            "turn_id": 1,
            "line_start": 4,
            "line_end": 6
        }
    ]
    """
    lines = raw_text.splitlines()
    turns: List[Dict] = []

    current_speaker = "unknown"
    buffer: List[str] = []
    turn_id = 0
    current_start_line = None
    current_end_line = None

    def flush():
        nonlocal turn_id, buffer, current_speaker, current_start_line, current_end_line
        text = "\n".join(buffer).strip()
        if text:
            turns.append({
                "speaker": current_speaker,
                "text": text,
                "turn_id": turn_id,
                "line_start": current_start_line,
                "line_end": current_end_line
            })
            turn_id += 1
        buffer = []
        current_start_line = None
        current_end_line = None

    for idx, line in enumerate(lines, start=1):
        line_stripped = line.strip()

        if not line_stripped:
            # mantenemos la línea vacía por estructura,
            # pero seguimos registrando rango si ya estamos dentro de turno
            if current_start_line is not None:
                current_end_line = idx
            buffer.append("")
            continue

        matched = False
        for pat, spk in SPEAKER_PATTERNS:
            if re.match(pat, line_stripped, flags=re.IGNORECASE):
                flush()
                current_speaker = spk
                current_start_line = idx
                current_end_line = idx

                cleaned = re.sub(pat, "", line_stripped, flags=re.IGNORECASE).strip()
                buffer.append(cleaned)
                matched = True
                break

        if matched:
            continue

        # Header lines like "Context: ..." are stored as meta blocks
        if current_speaker == "unknown" and any(
            line_stripped.lower().startswith(h) for h in HEADER_HINTS
        ):
            flush()
            current_speaker = "meta"
            current_start_line = idx
            current_end_line = idx
            buffer.append(line_stripped)
            continue

        if current_start_line is None:
            current_start_line = idx
        current_end_line = idx
        buffer.append(line_stripped)

    flush()
    return turns


# =========================
# CASE CHUNKING WITH LINES
# =========================
def chunk_turn_preserving_lines(
    turn_text: str,
    line_start: int,
    chunk_size: int = 450
) -> List[Dict]:
    """
    Chunk a turn while preserving approximate exact line ranges.

    We chunk by lines (not by arbitrary characters) so each chunk can keep:
    - text
    - line_start
    - line_end

    This is better for traceability than character-only splitting.
    """
    lines = turn_text.splitlines()
    if not lines:
        return []

    chunks: List[Dict] = []
    buffer: List[str] = []
    current_start = line_start
    current_end = line_start
    current_len = 0

    for i, ln in enumerate(lines):
        absolute_line = line_start + i
        ln_len = len(ln) + 1  # +1 for newline approximation

        # if adding this line exceeds chunk_size, flush current chunk first
        if buffer and current_len + ln_len > chunk_size:
            chunks.append({
                "text": "\n".join(buffer).strip(),
                "line_start": current_start,
                "line_end": current_end
            })
            buffer = [ln]
            current_start = absolute_line
            current_end = absolute_line
            current_len = ln_len
        else:
            if not buffer:
                current_start = absolute_line
            buffer.append(ln)
            current_end = absolute_line
            current_len += ln_len

    if buffer:
        chunks.append({
            "text": "\n".join(buffer).strip(),
            "line_start": current_start,
            "line_end": current_end
        })

    return [c for c in chunks if c["text"]]


# =========================
# IO / DB
# =========================
def load_text(path: Path) -> str:
    """Extract text with Unstructured (works for .txt and .pdf). Used mainly for manuals."""
    try:
        elements = partition(filename=str(path))
        text = "\n".join(
            e.text for e in elements if getattr(e, "text", None)
        )
        return text.strip()
    except Exception as e:
        print(f"⚠️ Error reading {path.name}: {e}")
        return ""


def load_case_text(path: Path) -> str:
    """
    Load case text preserving original line structure exactly.
    This is used for interviews (.txt), so line numbers remain meaningful.
    """
    try:
        return path.read_text(encoding="utf-8").strip()
    except UnicodeDecodeError:
        try:
            return path.read_text(encoding="latin-1").strip()
        except Exception as e:
            print(f"⚠️ Error reading case file {path.name}: {e}")
            return ""
    except Exception as e:
        print(f"⚠️ Error reading case file {path.name}: {e}")
        return ""


def get_db():
    return Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_PATH,
        embedding_function=EMB,
    )


def existing_doc_ids(db) -> set[str]:
    """Return existing doc_id values to avoid re-ingesting the same file."""
    try:
        data = db.get(include=["metadatas"])
        metadatas = data.get("metadatas", []) or []

        ids: set[str] = set()
        for m in metadatas:
            if isinstance(m, dict):
                doc_id = m.get("doc_id")
                if doc_id:
                    ids.add(doc_id)

        return ids

    except Exception as e:
        print(f"⚠️ Could not read existing doc_ids: {e}")
        return set()


def infer_layer(manual_path: Path) -> tuple[str, int]:
    """
    Given a manual located at:
    data/manuales/<layer_folder>/file.pdf

    return:
    (layer_name, layer_num)
    """
    layer_folder = manual_path.parent.name  # e.g. "01-clinical-guidelines"
    try:
        layer_num = int(layer_folder.split("-")[0])
    except Exception:
        layer_num = -1
    return layer_folder, layer_num


def add_document(db, text: str, meta_base: dict, base_id: str):
    """Split and add a document to Chroma (general use, e.g. manuals)."""
    chunks = splitter.split_text(text)
    if not chunks:
        return

    metadatas = []
    ids = []

    for i, _ in enumerate(chunks):
        m = dict(meta_base)
        m["chunk"] = i
        metadatas.append(m)
        ids.append(f"{base_id}::chunk{i}")

    db.add_texts(chunks, metadatas=metadatas, ids=ids)


def add_case_by_turns(db, full_text: str, meta_base: dict, base_id: str):
    """
    Ingest a case by splitting it into speaker turns (speaker + turn_id),
    preserving exact line ranges per chunk.
    """
    turns = split_into_turns_with_lines(full_text)

    for t in turns:
        speaker = t["speaker"]
        turn_id = t["turn_id"]
        turn_text = (t["text"] or "").strip()
        turn_line_start = t["line_start"]
        turn_line_end = t["line_end"]

        if not turn_text:
            continue

        # Optional: skip meta/unknown if you don't want them in retrieval
        if speaker in {"meta", "unknown"}:
            continue

        chunks = chunk_turn_preserving_lines(
            turn_text=turn_text,
            line_start=turn_line_start,
            chunk_size=450
        )

        if not chunks:
            continue

        metadatas = []
        ids = []

        for i, chunk_info in enumerate(chunks):
            m = dict(meta_base)
            m["speaker"] = speaker
            m["turn_id"] = turn_id
            m["chunk"] = i
            m["line_start"] = chunk_info["line_start"]
            m["line_end"] = chunk_info["line_end"]
            m["turn_line_start"] = turn_line_start
            m["turn_line_end"] = turn_line_end

            metadatas.append(m)
            ids.append(f"{base_id}::turn{turn_id}::{speaker}::chunk{i}")

        db.add_texts(
            [c["text"] for c in chunks],
            metadatas=metadatas,
            ids=ids
        )


# =========================
# MAIN
# =========================
def process_bulk():
    CASES_FOLDER.mkdir(parents=True, exist_ok=True)
    MANUALS_FOLDER.mkdir(parents=True, exist_ok=True)

    db = get_db()
    seen = existing_doc_ids(db)

    print("🔍 Analyzing existing database...")
    print(f"📋 Existing doc_ids: {len(seen)}")

    # ---- 1) CASES (.txt) ----
    case_files = sorted(
        [p for p in CASES_FOLDER.glob("*.txt") if p.is_file()]
    )

    if not case_files:
        print("📭 No .txt case files found in 'data/entradas/'.")
    else:
        for p in case_files:
            interview_id = p.stem
            doc_id = f"case::{interview_id}"

            if doc_id in seen:
                print(f"⏩ Skipping case (already exists): {p.name}")
                continue

            text = load_case_text(p)
            if not text:
                print(f"⚠️ Empty file: {p.name} (no extractable text?)")
                continue

            meta = {
                "type": "case",
                "interview_id": interview_id,
                "source": p.name,
                "rel_path": str(p.relative_to(Path("data"))),
                "doc_id": doc_id,
            }

            print(f"⚙️ Processing [CASE by turns + lines]: {p.name}")
            add_case_by_turns(db, text, meta, doc_id)

    # ---- 2) MANUALS (.pdf and .txt, recursive) ----
    manual_files = sorted([
        p for p in MANUALS_FOLDER.rglob("*")
        if p.is_file() and p.suffix.lower() in [".pdf", ".txt"]
    ])

    if not manual_files:
        print("📭 No manuals (.pdf/.txt) found in 'data/manuales/'.")
    else:
        for p in manual_files:
            layer_name, layer_num = infer_layer(p)
            doc_id = f"manual::{layer_name}::{p.stem}"

            if doc_id in seen:
                print(f"⏩ Skipping manual (already exists): {layer_name}/{p.name}")
                continue

            text = load_text(p)
            if not text:
                print(f"⚠️ Empty file: {layer_name}/{p.name} (if scanned PDF, OCR may be needed)")
                continue

            meta = {
                "type": "manual",
                "layer": layer_name,
                "layer_num": layer_num,
                "source": p.name,
                "rel_path": str(p.relative_to(Path("data"))),
                "doc_id": doc_id,
            }

            print(f"⚙️ Processing [MANUAL L{layer_num:02d}]: {layer_name}/{p.name}")
            add_document(db, text, meta, doc_id)

    try:
        db.persist()
    except Exception:
        pass

    print("\n✅ Ingestion finished.")
    print("ℹ️ Note: if you want new metadata (line_start/line_end/speaker/turn_id) to be reflected everywhere, delete data/chroma and re-ingest.")


if __name__ == "__main__":
    process_bulk()