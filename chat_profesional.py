import os
import re
import time
import warnings
from typing import Any, Dict, List, Optional, Tuple

# --- 1) CONFIG ---
os.environ["ANONYMIZED_TELEMETRY"] = "False"
warnings.filterwarnings("ignore")

# 🔎 DEBUG
DEBUG = os.getenv("CDSS_DEBUG", "0") == "1"

# Hugging Face
HF_TOKEN = os.getenv("HF_TOKEN")
HF_MODEL = "meta-llama/Llama-3.1-8B-Instruct"

# Retrieval config
MAX_CASE_TURNS = 3
CASE_CHUNK_K = 8
CASE_FETCH_K = 40
MANUAL_K = 1
MAX_HISTORY_ITEMS = 6

# --- 2) IMPORTS ---
from langchain_chroma import Chroma
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from huggingface_hub import InferenceClient
from langchain_core.documents import Document

# --- 3) PROMPT BASE ---
SYSTEM_RULES = """
You are a Clinical Decision Support Assistant (CDSS) intended to support professionals.
Your task is NOT to chat freely. Your task is to answer the QUESTION as directly, specifically, and faithfully as possible using only the retrieved INTERVIEW fragments.

PRIMARY OBJECTIVE:
Answer the specific question strictly from the INTERVIEW.
The MANUAL may only be used to name one rule or criterion, never as the main factual source about the case.

MANDATORY HARD RULES:
1) Use ONLY information that appears literally, or is directly and minimally deducible without clinical interpretation, in the INTERVIEW fragments.
2) DO NOT invent data.
3) DO NOT diagnose.
4) DO NOT infer gender, age, severity, intention, diagnosis, risk, background, or any attribute not explicitly stated in the INTERVIEW.
5) For questions about suicidal thoughts, self-harm, risk, safety, crisis, or urgent referral:
   - answer positively ONLY if it is explicitly stated in the INTERVIEW;
   - do NOT infer risk from sadness, distress, frustration, conflict, or vague suffering;
   - if it is not explicit, state clearly that it is not explicitly mentioned in the interview.
6) For questions about main concern:
   - answer with the most specific central concern supported by the INTERVIEW;
   - avoid vague summaries such as "multiple issues" unless the INTERVIEW itself is truly vague.
7) For questions about explicit emotions:
   - mention only emotions or affective states clearly expressed in the INTERVIEW;
   - do NOT convert general adjectives, interviewer explanations, personality traits, or ambiguous wording into explicit emotions of the patient.
8) For questions about behavior, pattern, or subjective experience:
   - describe only what appears in the INTERVIEW using brief observational language close to the text;
   - avoid unnecessary interpretive expressions such as "seems" or abstract clinical wording.
9) For recommendation or management questions:
   - distinguish between an explicit recommendation and a general reflection;
   - if there is no explicit recommendation, do not invent one.
10) Respect speaker labels. Do NOT attribute therapist/interviewer questions, reflections, or summaries to the patient. Use therapist text as evidence only if the question is explicitly about the therapist's recommendation, intervention, or wording.
11) DO NOT translate quotations. Quotes must be copied EXACTLY in the original language of the INTERVIEW.
12) "Evidence" must contain:
   - 1 or 2 EXACT textual quotations from the INTERVIEW, in quotation marks, if clear quotes exist;
   - or EXACTLY: Not mentioned in the interview.
13) The MANUAL may only be used for "Manual rule".
14) If there is no literal trigger in the INTERVIEW to apply a manual rule, write EXACTLY:
   Manual rule: Not applicable.
15) NEVER use the MANUAL as the main factual evidence for the case.
16) DO NOT repeat or copy the prompt or section names such as INTERVIEW, MANUAL, HISTORY, or QUESTION.
17) Write the "Response" line in natural, concise English.
18) Answer the QUESTION itself, not a generic summary of the interview.
19) ALWAYS return exactly 3 lines and nothing else.

DECISION POLICY:
- First determine whether the INTERVIEW answers the QUESTION clearly, partially, or not at all.
- If the INTERVIEW answers the question clearly, give a direct and specific answer.
- If the INTERVIEW answers only part of a multi-part question, say so briefly and only then mention the missing part.
- If the INTERVIEW does not contain sufficient direct information, write a direct absence answer aligned with the question, for example:
  - Response: Suicidal thoughts are not explicitly mentioned in the interview.
  - Response: Urgent referral is not explicitly mentioned in the interview.
  - Response: Explicit emotions are not clearly mentioned in the interview.
  Evidence: Not mentioned in the interview.
  Manual rule: Not applicable.

CONSISTENCY RULES:
- If Response says that something is not explicitly mentioned, Evidence must be EXACTLY:
  Not mentioned in the interview.
- If Evidence contains a real textual quotation, Response must not say that the information is not mentioned.
- Prefer a question-aligned answer over a vague summary.
- Keep the response brief, but as specific as the interview allows.

FINAL REQUIRED FORMAT:
Response: ...
Evidence: ...
Manual rule: ...
""".strip()

PROMPT_TEMPLATE = """
TASK:
{task_hint}

DECISION INSTRUCTIONS:
- First decide whether the INTERVIEW answers the question clearly, partially, or not at all.
- Then write ONLY the final 3-line output.
- Do not explain your reasoning.
- The interview context is grouped by DIALOGUE TURN and SPEAKER. Keep speaker attribution faithful.
- Answer the QUESTION itself, not a generic summary.
- When the question asks about the presence, absence, or mention of something, make the Response explicitly about that target concept.

Interview context:
{case_context}

Manual context:
{manual_context}

Brief history:
{chat_history}

Question:
{question}
""".strip()

# --- 4) GLOBAL CACHE ---
_EMBEDDINGS = None
_VECTORSTORE = None
_HF_CLIENT = None


# --- 5) LOW-LEVEL HELPERS ---

def shorten(text: str, max_chars: int = 800) -> str:
    text = " ".join((text or "").split())
    return text[:max_chars] + ("…" if len(text) > max_chars else "")


def normalize_ws(text: str) -> str:
    return " ".join((text or "").split()).strip()


def safe_int(value: Any, default: int = 10**9) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def format_line_range(start_line: Optional[int], end_line: Optional[int]) -> str:
    if start_line is None and end_line is None:
        return ""
    if start_line is None:
        return str(end_line)
    if end_line is None:
        return str(start_line)
    if start_line == end_line:
        return str(start_line)
    return f"{start_line}-{end_line}"


def format_history(history: Optional[List[Tuple[str, str]]]) -> str:
    if not history:
        return "(empty)"
    lines = []
    for i, (q, a) in enumerate(history[-2:], start=1):
        a_short = normalize_ws(a)
        if len(a_short) > 250:
            a_short = a_short[:250] + "…"
        lines.append(f"- Q{i}: {q}")
        lines.append(f"  A{i}: {a_short}")
    return "\n".join(lines)


def trim_history(history: Optional[List[Tuple[str, str]]]) -> List[Tuple[str, str]]:
    if not history:
        return []
    return history[-MAX_HISTORY_ITEMS:]


def looks_like_guideline_question(q: str) -> bool:
    ql = q.lower()
    keywords = [
        "guideline", "guidelines", "mhgap", "nice",
        "treatment", "intervention", "refer", "referral",
        "management", "plan", "recommend", "recommendation",
        "criterion", "criteria", "protocol", "therapy", "therapeutic"
    ]
    return any(k in ql for k in keywords)


def looks_like_emotion_pattern_question(q: str) -> bool:
    ql = q.lower()
    keywords = [
        "emotion", "emotions", "emotional", "anger", "fear", "frustration",
        "lie", "lying", "deceive", "manipulate", "pattern",
        "validation", "oars", "motivational"
    ]
    return any(k in ql for k in keywords)


def looks_like_risk_question(q: str) -> bool:
    ql = q.lower()
    keywords = [
        "suicidal", "suicide", "self-harm", "self harm", "self injury",
        "risk", "danger", "safety", "urgent referral", "urgent",
        "emergency", "crisis", "harm himself", "harm herself",
        "harm themselves"
    ]
    return any(k in ql for k in keywords)


def looks_like_main_concern_question(q: str) -> bool:
    ql = q.lower()
    keywords = [
        "main concern", "primary concern", "chief complaint",
        "main issue", "main problem", "what is the patient worried about",
        "what concerns the patient most"
    ]
    return any(k in ql for k in keywords)


def classify_task_hint(question: str) -> str:
    q = question.lower()

    if looks_like_risk_question(question):
        return (
            "Answer the risk-related question directly and conservatively. "
            "Only state suicidal thoughts, self-harm risk, safety risk, crisis, or urgent referral if this is explicitly mentioned in the interview. "
            "Do not infer risk from general suffering, sadness, frustration, conflict, or distress. "
            "If the target concept is not explicit, write a direct absence answer naming that concept, such as "
            "'Suicidal thoughts are not explicitly mentioned in the interview.'"
        )

    if looks_like_main_concern_question(question):
        return (
            "State the patient's main concern as specifically as the interview allows, in one short sentence. "
            "Prefer the patient's concrete problem over vague wording like 'multiple issues'. "
            "Do not broaden the answer beyond the central concern supported by the interview."
        )

    if any(k in q for k in [
        "recommend", "recommended", "management", "manage",
        "what should", "intervention", "plan", "strategy", "strategies"
    ]):
        return (
            "Identify only recommendations, guidance, or suggestions explicitly mentioned in the interview. "
            "Distinguish between an explicit recommendation and a general reflection. "
            "If there is no explicit recommendation, do not invent one. "
            "If there is only general guidance from the interviewer, state that cautiously and briefly."
        )

    if any(k in q for k in [
        "emotion", "emotions", "emotional", "fear", "anger", "frustration"
    ]):
        return (
            "Extract only explicit emotions or affective states mentioned in the interview. "
            "Do not turn general explanations, personal traits, interviewer reflections, or ambiguous wording into explicit emotions of the patient. "
            "If only one clear emotion appears, mention only that one. "
            "If explicit emotions are not clearly mentioned, say so directly."
        )

    if any(k in q for k in [
        "behavior", "behaviour", "pattern", "what does the patient describe",
        "describe the patient", "clinical signs", "indicators", "subjective experience"
    ]):
        return (
            "Describe only what appears in the interview using observational, brief, natural English. "
            "Use wording close to the text. "
            "Avoid awkward literal wording and avoid non-explicit clinical interpretations."
        )

    return (
        "Answer the specific question directly using only the interview. "
        "Prefer a precise, question-aligned answer over a generic summary. "
        "Distinguish between clear evidence, partial evidence, and absence of evidence."
    )


# --- 6) BACKEND INITIALIZATION ---

def get_embeddings():
    global _EMBEDDINGS
    if _EMBEDDINGS is None:
        _EMBEDDINGS = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")
    return _EMBEDDINGS


def get_vectorstore() -> Chroma:
    global _VECTORSTORE
    if _VECTORSTORE is None:
        chroma_path = "data/chroma"
        if not os.path.exists(chroma_path):
            raise FileNotFoundError("data/chroma does not exist. Run: python ingest_bulk.py")

        _VECTORSTORE = Chroma(
            collection_name="interviews",
            embedding_function=get_embeddings(),
            persist_directory=chroma_path
        )
    return _VECTORSTORE


def get_hf_client() -> InferenceClient:
    global _HF_CLIENT
    if _HF_CLIENT is None:
        if not HF_TOKEN:
            raise ValueError('Missing HF_TOKEN environment variable. Example: $env:HF_TOKEN="your_token_here"')
        _HF_CLIENT = InferenceClient(api_key=HF_TOKEN)
    return _HF_CLIENT


def init_backend() -> Dict[str, str]:
    """
    Útil para Streamlit: fuerza la inicialización y permite detectar errores pronto.
    """
    get_vectorstore()
    get_hf_client()
    return {
        "status": "ok",
        "hf_model": HF_MODEL,
        "vectorstore_collection": "interviews",
    }


# --- 7) DOCUMENT / TURN HELPERS ---

def build_context(docs: List[Document], max_docs: int) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Contexto genérico para manuales u otros docs no conversacionales.
    """
    if not docs:
        return "(empty)", []

    parts = []
    items = []

    for d in docs[:max_docs]:
        meta = d.metadata or {}
        src = meta.get("source", "unknown")
        layer = meta.get("layer", "")
        layer_num = meta.get("layer_num", "")
        dtype = meta.get("type", "general")

        label = f"[{str(dtype).upper()}]"
        if layer:
            label += f"[{layer}]"
        if layer_num != "":
            label += f"[L{layer_num}]"

        excerpt = shorten(d.page_content, 700)
        parts.append(f"{label} {src}:\n{excerpt}\n")
        items.append({
            "source": src,
            "type": dtype,
            "layer": layer,
            "layer_num": layer_num,
            "excerpt": excerpt,
            "label": label,
        })

    return "\n".join(parts).strip(), items


def extract_turn_ids(docs: List[Document], max_turns: int = MAX_CASE_TURNS) -> List[Any]:
    turn_ids = []
    seen = set()

    for d in docs:
        meta = d.metadata or {}
        turn_id = meta.get("turn_id")
        if turn_id is None:
            continue

        key = str(turn_id)
        if key in seen:
            continue

        seen.add(key)
        turn_ids.append(turn_id)

        if len(turn_ids) >= max_turns:
            break

    return turn_ids


def fetch_all_case_docs(vs: Chroma, interview_id: str) -> List[Document]:
    raw = vs.get(
        where={
            "$and": [
                {"interview_id": {"$eq": interview_id}},
                {"type": {"$eq": "case"}},
            ]
        },
        include=["documents", "metadatas"]
    )

    documents = raw.get("documents") or []
    metadatas = raw.get("metadatas") or []

    docs = []
    for text, meta in zip(documents, metadatas):
        docs.append(Document(page_content=text or "", metadata=meta or {}))
    return docs


def fetch_case_docs_for_turns(vs: Chroma, interview_id: str, turn_ids: List[Any]) -> List[Document]:
    if not turn_ids:
        return []

    raw = vs.get(
        where={
            "$and": [
                {"interview_id": {"$eq": interview_id}},
                {"type": {"$eq": "case"}},
                {"turn_id": {"$in": turn_ids}},
            ]
        },
        include=["documents", "metadatas"]
    )

    documents = raw.get("documents") or []
    metadatas = raw.get("metadatas") or []

    docs = []
    for text, meta in zip(documents, metadatas):
        docs.append(Document(page_content=text or "", metadata=meta or {}))

    return docs


def group_docs_by_turn(
    docs: List[Document],
    preferred_order: Optional[List[Any]] = None
) -> List[Dict[str, Any]]:
    """
    Agrupa chunks por turn_id y reconstruye el texto del turno ordenando por line_start.
    """
    if not docs:
        return []

    groups: Dict[str, Dict[str, Any]] = {}

    for d in docs:
        meta = d.metadata or {}
        turn_id = meta.get("turn_id")
        key = str(turn_id) if turn_id is not None else f"fallback::{meta.get('source', 'unknown')}::{meta.get('chunk', '')}"

        if key not in groups:
            groups[key] = {
                "turn_id": turn_id if turn_id is not None else "NA",
                "speaker": meta.get("speaker", "unknown"),
                "source": meta.get("source", "unknown"),
                "docs": [],
            }

        if groups[key]["speaker"] in ("", None, "unknown"):
            groups[key]["speaker"] = meta.get("speaker", "unknown")
        if groups[key]["source"] in ("", None, "unknown"):
            groups[key]["source"] = meta.get("source", "unknown")

        groups[key]["docs"].append(d)

    order_map = {}
    if preferred_order:
        order_map = {str(turn_id): i for i, turn_id in enumerate(preferred_order)}

    def group_sort_key(group_key: str):
        group = groups[group_key]
        turn_id = group["turn_id"]
        if order_map:
            return (order_map.get(str(turn_id), 10**9), safe_int(turn_id))
        return (safe_int(turn_id),)

    ordered_keys = sorted(groups.keys(), key=group_sort_key)

    turns = []
    for key in ordered_keys:
        group = groups[key]
        group_docs = sorted(
            group["docs"],
            key=lambda d: (
                safe_int((d.metadata or {}).get("line_start")),
                safe_int((d.metadata or {}).get("chunk")),
            )
        )

        text_parts = []
        seen_texts = set()
        line_starts = []
        line_ends = []
        turn_line_starts = []
        turn_line_ends = []

        for d in group_docs:
            meta = d.metadata or {}
            chunk_text = normalize_ws(d.page_content)

            if chunk_text and chunk_text not in seen_texts:
                text_parts.append(chunk_text)
                seen_texts.add(chunk_text)

            if meta.get("line_start") is not None:
                line_starts.append(safe_int(meta.get("line_start")))
            if meta.get("line_end") is not None:
                line_ends.append(safe_int(meta.get("line_end")))
            if meta.get("turn_line_start") is not None:
                turn_line_starts.append(safe_int(meta.get("turn_line_start")))
            if meta.get("turn_line_end") is not None:
                turn_line_ends.append(safe_int(meta.get("turn_line_end")))

        start_line = min(turn_line_starts) if turn_line_starts else (min(line_starts) if line_starts else None)
        end_line = max(turn_line_ends) if turn_line_ends else (max(line_ends) if line_ends else None)

        turns.append({
            "turn_id": group["turn_id"],
            "speaker": group["speaker"] or "unknown",
            "source": group["source"] or "unknown",
            "start_line": start_line,
            "end_line": end_line,
            "lines": format_line_range(start_line, end_line),
            "text": " ".join(text_parts).strip(),
            "num_chunks": len(group_docs),
        })

    return turns


def build_case_turn_context(
    seed_docs: List[Document],
    full_turn_docs: List[Document],
    max_turns: int = MAX_CASE_TURNS
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Construye el contexto del caso por turnos completos y hablantes.
    """
    if not seed_docs:
        return "(empty)", []

    turn_ids = extract_turn_ids(seed_docs, max_turns=max_turns)

    if not turn_ids or not full_turn_docs:
        flat_parts = []
        fallback_turns = []
        for i, d in enumerate(seed_docs[:max_turns], start=1):
            meta = d.metadata or {}
            src = meta.get("source", "unknown")
            speaker = meta.get("speaker", "unknown")
            turn_id = meta.get("turn_id", f"chunk-{i}")
            start_line = meta.get("line_start")
            end_line = meta.get("line_end")

            text = normalize_ws(d.page_content)
            label = f"[CASE][TURN {turn_id}][SPEAKER: {str(speaker).upper()}]"
            if start_line is not None or end_line is not None:
                label += f"[LINES {format_line_range(start_line, end_line)}]"

            flat_parts.append(f"{label} {src}:\n{shorten(text, 1400)}\n")
            fallback_turns.append({
                "turn_id": turn_id,
                "speaker": speaker,
                "source": src,
                "start_line": start_line,
                "end_line": end_line,
                "lines": format_line_range(start_line, end_line),
                "text": text,
                "num_chunks": 1,
            })

        return "\n".join(flat_parts).strip() if flat_parts else "(empty)", fallback_turns

    turns = group_docs_by_turn(full_turn_docs, preferred_order=turn_ids)
    wanted = {str(tid) for tid in turn_ids}
    turns = [t for t in turns if str(t["turn_id"]) in wanted][:max_turns]

    parts = []
    for turn in turns:
        label = f"[CASE][TURN {turn['turn_id']}][SPEAKER: {str(turn['speaker']).upper()}]"
        if turn["lines"]:
            label += f"[LINES {turn['lines']}]"
        parts.append(f"{label} {turn['source']}:\n{shorten(turn['text'], 1400)}\n")

    if not parts:
        return "(empty)", []

    return "\n".join(parts).strip(), turns


# --- 8) ANSWER / EVIDENCE HELPERS ---

def validate_answer_format(answer_str: str) -> bool:
    lines = [ln.strip() for ln in answer_str.splitlines() if ln.strip()]
    if len(lines) < 3:
        return False
    ok1 = lines[0].startswith("Response:")
    ok2 = lines[1].startswith("Evidence:")
    ok3 = lines[2].startswith("Manual rule:")
    return ok1 and ok2 and ok3


def parse_answer(answer_str: str) -> Dict[str, str]:
    response = ""
    evidence = ""
    manual_rule = ""

    lines = [ln.strip() for ln in answer_str.splitlines() if ln.strip()]
    for ln in lines:
        if ln.startswith("Response:"):
            response = ln.replace("Response:", "", 1).strip()
        elif ln.startswith("Evidence:"):
            evidence = ln.replace("Evidence:", "", 1).strip()
        elif ln.startswith("Manual rule:"):
            manual_rule = ln.replace("Manual rule:", "", 1).strip()

    if not response and lines:
        response = lines[0]
    if not evidence:
        evidence = "Not mentioned in the interview."
    if not manual_rule:
        manual_rule = "Not applicable."

    return {
        "response": response or "Not mentioned in the interview.",
        "evidence": evidence,
        "manual_rule": manual_rule,
    }


def extract_quotes(text: str) -> List[str]:
    if not text:
        return []

    matches = re.findall(r'["“](.+?)["”]', text)
    quotes = []
    seen = set()

    for q in matches:
        q_norm = normalize_ws(q)
        if q_norm and q_norm not in seen:
            quotes.append(q_norm)
            seen.add(q_norm)

    return quotes


def match_quote_to_turn(quote: str, turns: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    q_norm = normalize_ws(quote).lower()
    if not q_norm:
        return None

    for turn in turns:
        text_norm = normalize_ws(turn.get("text", "")).lower()
        if q_norm in text_norm:
            return turn
    return None


def build_evidence_items(evidence_text: str, retrieved_turns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    quotes = extract_quotes(evidence_text)
    items = []

    for i, quote in enumerate(quotes, start=1):
        matched_turn = match_quote_to_turn(quote, retrieved_turns)

        if matched_turn:
            items.append({
                "ref": i,
                "source": matched_turn.get("source", "unknown"),
                "speaker": matched_turn.get("speaker", "unknown"),
                "turn_id": matched_turn.get("turn_id"),
                "lines": matched_turn.get("lines", ""),
                "quote": quote,
            })
        else:
            items.append({
                "ref": i,
                "source": "unknown",
                "speaker": "unknown",
                "turn_id": None,
                "lines": "",
                "quote": quote,
            })

    return items


def add_refs_to_answer(answer: str, evidence_items: List[Dict[str, Any]]) -> str:
    if not answer:
        return ""
    if not evidence_items:
        return answer

    refs = " ".join(f"[{item['ref']}]" for item in evidence_items)
    return f"{answer} {refs}".strip()


def call_hf_model(client: InferenceClient, prompt: str) -> str:
    response = client.chat_completion(
        model=HF_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_RULES},
            {"role": "user", "content": prompt},
        ],
        max_tokens=300,
        temperature=0.0,
    )

    content = response.choices[0].message.content

    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
            else:
                parts.append(str(item))
        return "\n".join(parts).strip()

    return str(content).strip()


# --- 9) PUBLIC FUNCTIONS FOR STREAMLIT / APP ---

def list_interviews() -> List[Dict[str, Any]]:
    """
    Devuelve un catálogo simple de entrevistas basado en metadatos ya indexados en Chroma.
    """
    vs = get_vectorstore()

    raw = vs.get(
        where={"type": {"$eq": "case"}},
        include=["metadatas"]
    )

    metadatas = raw.get("metadatas") or []
    groups: Dict[str, Dict[str, Any]] = {}

    for meta in metadatas:
        meta = meta or {}
        interview_id = meta.get("interview_id")
        if not interview_id:
            continue

        if interview_id not in groups:
            groups[interview_id] = {
                "interview_id": interview_id,
                "sources": set(),
                "speakers": set(),
                "turn_ids": set(),
                "num_chunks": 0,
            }

        item = groups[interview_id]
        item["num_chunks"] += 1

        src = meta.get("source")
        if src:
            item["sources"].add(src)

        speaker = meta.get("speaker")
        if speaker:
            item["speakers"].add(str(speaker))

        turn_id = meta.get("turn_id")
        if turn_id is not None:
            item["turn_ids"].add(str(turn_id))

    results = []
    for interview_id, item in groups.items():
        sources = sorted(item["sources"])
        speakers = sorted(item["speakers"])

        results.append({
            "interview_id": interview_id,
            "source": sources[0] if sources else "unknown",
            "sources": sources,
            "speakers": speakers,
            "num_turns": len(item["turn_ids"]),
            "num_chunks": item["num_chunks"],
        })

    results.sort(key=lambda x: x["interview_id"])
    return results


def get_interview_preview(interview_id: str, max_preview_turns: int = 3) -> Dict[str, Any]:
    """
    Devuelve un preview estructurado del caso.
    """
    if not interview_id:
        raise ValueError("interview_id is required")

    vs = get_vectorstore()
    docs = fetch_all_case_docs(vs, interview_id)
    turns = group_docs_by_turn(docs)

    if not turns:
        return {
            "interview_id": interview_id,
            "found": False,
            "source": "unknown",
            "speakers": [],
            "num_turns": 0,
            "preview_turns": [],
            "preview_text": "(empty)",
        }

    speakers = sorted({str(t.get("speaker", "unknown")) for t in turns if t.get("speaker")})
    sources = sorted({str(t.get("source", "unknown")) for t in turns if t.get("source")})

    preview_turns = []
    for turn in turns[:max_preview_turns]:
        preview_turns.append({
            "turn_id": turn["turn_id"],
            "speaker": turn["speaker"],
            "source": turn["source"],
            "lines": turn["lines"],
            "text": shorten(turn["text"], 400),
        })

    preview_blocks = []
    for turn in preview_turns:
        block = f"[TURN {turn['turn_id']}][{str(turn['speaker']).upper()}]"
        if turn["lines"]:
            block += f"[LINES {turn['lines']}]"
        block += f" {turn['text']}"
        preview_blocks.append(block)

    return {
        "interview_id": interview_id,
        "found": True,
        "source": sources[0] if sources else "unknown",
        "sources": sources,
        "speakers": speakers,
        "num_turns": len(turns),
        "preview_turns": preview_turns,
        "preview_text": "\n\n".join(preview_blocks),
    }


def ask_case(
    question: str,
    interview_id: str,
    history: Optional[List[Tuple[str, str]]] = None,
    thread_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Función principal para usar desde Streamlit.
    Devuelve una estructura rica, no solo texto.
    """
    if not question or not question.strip():
        raise ValueError("question is required")
    if not interview_id or not interview_id.strip():
        raise ValueError("interview_id is required")

    vs = get_vectorstore()
    client = get_hf_client()
    history = trim_history(history)

    t0 = time.time()

    # 1) Case retriever: chunks semánticos para detectar turnos relevantes
    retriever_case = vs.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": CASE_CHUNK_K,
            "fetch_k": CASE_FETCH_K,
            "filter": {
                "$and": [
                    {"interview_id": {"$eq": interview_id}},
                    {"type": {"$eq": "case"}}
                ]
            }
        }
    )

    case_seed_docs = retriever_case.invoke(question)

    # 2) Expandir a turnos completos
    turn_ids = extract_turn_ids(case_seed_docs, max_turns=MAX_CASE_TURNS)
    case_turn_docs = fetch_case_docs_for_turns(vs, interview_id, turn_ids)
    case_context, case_turns = build_case_turn_context(
        case_seed_docs,
        case_turn_docs,
        max_turns=MAX_CASE_TURNS
    )

    # 3) Decidir capas de manual dinámicamente
    layers = [0, 2]
    if looks_like_guideline_question(question):
        layers.append(1)
    if looks_like_emotion_pattern_question(question):
        layers.append(3)

    # 4) Manual retriever
    retriever_manual = vs.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": MANUAL_K,
            "fetch_k": 20,
            "filter": {
                "$and": [
                    {"type": {"$eq": "manual"}},
                    {"layer_num": {"$in": layers}}
                ]
            }
        }
    )
    manual_docs = retriever_manual.invoke(question)
    manual_context, manual_sources = build_context(manual_docs, max_docs=1)

    # 5) Prompt
    task_hint = classify_task_hint(question)
    prompt = PROMPT_TEMPLATE.format(
        task_hint=task_hint,
        chat_history=format_history(history),
        manual_context=manual_context,
        case_context=case_context,
        question=question
    )

    if question not in prompt:
        raise RuntimeError("The question is NOT in the final prompt. Formatting/variable bug.")

    if DEBUG:
        print("\n==================== 🔎 DEBUG ====================")
        print("QUESTION:", question)
        print("INTERVIEW ID:", interview_id)
        print("THREAD ID:", thread_id)
        print("TASK_HINT:", task_hint)
        print("TURN IDS:", turn_ids)
        print("\n--- CASE CONTEXT ---")
        print(case_context[:1800])
        print("\n--- MANUAL CONTEXT ---")
        print(manual_context[:1200])
        print("\n--- FINAL PROMPT (start) ---")
        print(prompt[:2200])
        print("--- FINAL PROMPT (end) ---")
        print(prompt[-1200:])
        print("==================================================\n")

    # 6) LLM call
    raw_answer = call_hf_model(client, prompt)
    parsed = parse_answer(raw_answer)
    evidence_items = build_evidence_items(parsed["evidence"], case_turns)
    answer_with_refs = add_refs_to_answer(parsed["response"], evidence_items)

    t1 = time.time()

    next_history = trim_history(history + [(question, raw_answer)])

    return {
        "thread_id": thread_id,
        "interview_id": interview_id,
        "question": question,
        "answer": parsed["response"],
        "answer_with_refs": answer_with_refs,
        "evidence_text": parsed["evidence"],
        "manual_rule": parsed["manual_rule"],
        "raw_answer": raw_answer,
        "format_ok": validate_answer_format(raw_answer),
        "evidence": evidence_items,
        "manual_sources": manual_sources,
        "retrieved_turns": case_turns,
        "retrieved_turn_ids": [t.get("turn_id") for t in case_turns],
        "history_used": history,
        "next_history": next_history,
        "elapsed_sec": round(t1 - t0, 3),
    }


# --- 10) OPTIONAL DEBUG HELPERS ---

def debug_print_docs(title: str, docs: List[Document]):
    print(f"\n🔎 DEBUG {title}: {len(docs) if docs else 0} docs")
    if not docs:
        return

    for i, d in enumerate(docs[:3], start=1):
        meta = d.metadata or {}
        src = meta.get("source", "unknown")
        dtype = meta.get("type", "general")
        turn_id = meta.get("turn_id", "NA")
        speaker = meta.get("speaker", "unknown")
        print(f"  - {i}) type={dtype} turn={turn_id} speaker={speaker} source={src}")
        print(f"     snippet: {shorten(d.page_content, 220)}")


def debug_print_turns(title: str, turns: List[Dict[str, Any]]):
    print(f"\n🔎 DEBUG {title}: {len(turns) if turns else 0} turns")
    if not turns:
        return

    for i, t in enumerate(turns[:3], start=1):
        print(
            f"  - {i}) turn={t['turn_id']} speaker={t['speaker']} "
            f"lines={t['lines']} chunks={t['num_chunks']} source={t['source']}"
        )
        print(f"     text: {shorten(t['text'], 260)}")


# --- 11) CLI MODE FOR MANUAL TESTING ---

def main():
    print("\n=== CDSS: CLINICAL DECISION SUPPORT SYSTEM (backend reusable + CLI) ===")

    try:
        info = init_backend()
    except Exception as e:
        print(f"❌ Init error: {e}")
        return

    print("✅ System ready.")
    print(f"🤖 Model: {info['hf_model']}")

    interview_id = input("📂 Patient ID (e.g. 'chat_random_564'): ").strip()
    if not interview_id:
        print("❌ Invalid ID.")
        return

    history: List[Tuple[str, str]] = []

    print(f"🔒 Case: {interview_id}")
    print("🧩 Retrieval mode: semantic search → relevant turns → full turn reconstruction by speaker.")
    print("📚 Manuals: default layers 0 and 2. Layer 1 or 3 only if the question requires it.")
    print("\n💬 Chat started. Type 'exit' to quit.")

    while True:
        try:
            question = input("\n👨‍⚕️ Question: ").strip()
            if not question:
                continue
            if question.lower() in ["exit", "quit"]:
                break

            result = ask_case(
                question=question,
                interview_id=interview_id,
                history=history,
                thread_id="cli-thread"
            )

            history = result["next_history"]

            print("\n" + "—" * 60)
            print(f"Response: {result['answer']}")
            print(f"Evidence: {result['evidence_text']}")
            print(f"Manual rule: {result['manual_rule']}")
            print("—" * 60)

            print(f"📎 Display: {result['answer_with_refs']}")
            print(f"📚 Evidence items: {len(result['evidence'])}")
            for item in result["evidence"]:
                print(
                    f"   [{item['ref']}] source={item['source']} | speaker={item['speaker']} "
                    f"| turn={item['turn_id']} | lines={item['lines']}"
                )
                print(f'       "{item["quote"]}"')

            if result["manual_sources"]:
                print("📘 Manual sources:")
                for ms in result["manual_sources"]:
                    print(
                        f"   • {ms['label']} {ms['source']} "
                        f"(layer={ms['layer_num']})"
                    )

            print(f"⏱️ {result['elapsed_sec']:.2f}s")

            if not result["format_ok"]:
                print("⚠️ WARNING: The model did not follow the 3-line format exactly.")

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"❌ Chat error: {e}")


if __name__ == "__main__":
    main()