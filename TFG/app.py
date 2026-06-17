import html
import os
import re
import time
import warnings
from math import ceil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

os.environ["ANONYMIZED_TELEMETRY"] = "False"
warnings.filterwarnings("ignore")

from chat_profesional import (
    HF_MODEL,
    init_backend,
    list_interviews,
    get_interview_preview,
    ask_case,
)

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="CDSS · Clinical Decision Support",
    page_icon="⚕",
    layout="wide",
)

PAGE_SIZE = 10

NAV_VIEWS = [
    "Interviews",
    "Clinical Chat",
    "RAG Evaluation",
    "Admin / Debug",
]

EXAMPLE_QUESTIONS = [
    "What is the patient's main concern?",
    "Are suicidal thoughts mentioned in the interview?",
    "What explicit emotions are mentioned in the interview?",
    "What behavior or subjective experience does the patient describe?",
    "Is there any mention of risk, self-harm, or urgent referral?",
]

SUMMARY_QUESTION = """
Provide a concise clinical summary of this interview.

Requirements:
- Use only information explicitly present in the interview.
- Do not diagnose.
- Do not infer suicidal ideation or self-harm unless explicitly mentioned.
- Structure the answer with:
  1. Brief summary
  2. Main concerns
  3. Explicit emotions
  4. Relevant behaviors or subjective experiences
  5. Risk-related mentions, only if explicit
  6. Suggested clinical keywords
"""

KEYWORDS_QUESTION = """
Extract the most relevant clinical keywords from this interview.

Requirements:
- Use only explicit content from the interview.
- Return a short comma-separated list.
- Include symptoms, emotions, behaviors, risks, or relevant clinical themes.
- Do not infer diagnoses.
"""

SORT_OPTIONS = [
    "Case ID · A to Z",
    "Case ID · Z to A",
    "Most turns first",
    "Most chunks first",
]

DEFAULT_USERS = {
    "admin": {
        "password": "admin123",
        "role": "Administrator",
        "display_name": "System Administrator",
        "status": "Active",
    },
    "therapist": {
        "password": "therapist123",
        "role": "Therapist",
        "display_name": "Clinical Therapist",
        "status": "Active",
    },
    "therapist2": {
        "password": "therapist2123",
        "role": "Therapist",
        "display_name": "Therapist Two",
        "status": "Active",
    },
    "therapist3": {
        "password": "therapist3123",
        "role": "Therapist",
        "display_name": "Therapist Three",
        "status": "Active",
    },
}

ROLE_NAVIGATION = {
    "Administrator": [
        "Interviews",
        "Clinical Chat",
        "RAG Evaluation",
        "Admin / Debug",
    ],
    "Therapist": [
        "Interviews",
        "Clinical Chat",
    ],
}

ADMIN_ONLY_VIEWS = {
    "RAG Evaluation",
    "Admin / Debug",
}


# ============================================================
# STYLE SYSTEM
# ============================================================
def inject_custom_css() -> None:
    st.markdown(
        """
        <style>
        :root {
            --bg: #fafafa;
            --surface: #ffffff;
            --surface-muted: #f6f8fb;
            --border: #e5e7eb;
            --border-strong: #d1d5db;
            --text: #111827;
            --text-soft: #64748b;
            --text-muted: #94a3b8;
            --accent: #0f6fec;
            --accent-hover: #0b5ed7;
            --accent-soft: #eff6ff;
            --danger: #b42318;
            --danger-soft: #fef3f2;
            --radius-sm: 8px;
            --radius-md: 12px;
            --radius-lg: 18px;
            --shadow-sm: 0 1px 2px rgba(15, 23, 42, 0.05);
            --shadow-md: 0 8px 24px rgba(15, 23, 42, 0.08);
        }

        .stApp {
            background: var(--bg);
            color: var(--text);
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Inter, system-ui, sans-serif;
        }

        [data-testid="stSidebar"] {
            background: var(--surface);
            border-right: 1px solid var(--border);
        }

        h1, h2, h3 {
            color: var(--text);
            letter-spacing: -0.02em;
        }

        h1 {
            font-size: 1.65rem !important;
            font-weight: 720 !important;
        }

        h2 {
            font-size: 1.25rem !important;
            font-weight: 680 !important;
        }

        h3 {
            font-size: 1rem !important;
            font-weight: 650 !important;
        }

        div.stButton > button {
            border-radius: var(--radius-md) !important;
            min-height: 2.35rem !important;
            font-size: 0.88rem !important;
            font-weight: 600 !important;
            transition: transform 0.12s ease, box-shadow 0.12s ease, border-color 0.12s ease, background 0.12s ease !important;
        }

        div.stButton > button:hover:not(:disabled) {
            transform: translateY(-1px);
        }

        div.stButton > button[kind="primary"] {
            background: var(--accent) !important;
            color: #ffffff !important;
            border: 1px solid var(--accent) !important;
            box-shadow: 0 4px 12px rgba(15, 111, 236, 0.20) !important;
        }

        div.stButton > button[kind="primary"]:hover:not(:disabled) {
            background: var(--accent-hover) !important;
            border-color: var(--accent-hover) !important;
            box-shadow: 0 8px 18px rgba(15, 111, 236, 0.22) !important;
        }

        div.stButton > button[kind="secondary"] {
            background: var(--surface) !important;
            color: var(--text) !important;
            border: 1px solid var(--border) !important;
            box-shadow: var(--shadow-sm) !important;
        }

        div.stButton > button[kind="secondary"]:hover:not(:disabled) {
            background: var(--surface-muted) !important;
            border-color: var(--border-strong) !important;
        }

        div.stButton > button:disabled {
            background: var(--surface-muted) !important;
            color: var(--text-muted) !important;
            border: 1px solid var(--border) !important;
            box-shadow: none !important;
            transform: none !important;
        }

        [data-testid="stTextInput"] input,
        [data-testid="stNumberInput"] input,
        [data-testid="stSelectbox"] div[data-baseweb="select"] > div {
            border-radius: var(--radius-md) !important;
            border-color: var(--border) !important;
            font-size: 0.9rem !important;
        }

        [data-testid="stExpander"] {
            border: 1px solid var(--border) !important;
            border-radius: var(--radius-md) !important;
            background: var(--surface) !important;
        }

        [data-testid="stChatInput"] {
            position: sticky;
            bottom: 0;
            z-index: 100;
            background: linear-gradient(to bottom, rgba(250,250,250,0), rgba(250,250,250,0.96) 36%, var(--bg));
            padding-top: 1rem;
            padding-bottom: 0.5rem;
        }

        .app-header {
            margin-bottom: 1.15rem;
        }

        .app-title {
            font-size: 1.65rem;
            font-weight: 760;
            color: var(--text);
            letter-spacing: -0.035em;
            line-height: 1.15;
        }

        .app-subtitle {
            color: var(--text-soft);
            font-size: 0.9rem;
            margin-top: 0.25rem;
        }

        .nav-shell {
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: var(--radius-lg);
            padding: 0.45rem;
            box-shadow: var(--shadow-sm);
            margin-bottom: 1.1rem;
        }

        .section-label {
            color: var(--text-muted);
            font-size: 0.72rem;
            font-weight: 760;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            margin-bottom: 0.5rem;
        }

        .muted-copy {
            color: var(--text-soft);
            font-size: 0.9rem;
            line-height: 1.45;
        }

        .interview-card {
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: var(--radius-lg);
            padding: 1.05rem 1.15rem;
            margin-bottom: 0.75rem;
            box-shadow: var(--shadow-sm);
            transition: border-color 0.12s ease, box-shadow 0.12s ease, transform 0.12s ease;
        }

        .interview-card:hover {
            border-color: var(--border-strong);
            box-shadow: var(--shadow-md);
            transform: translateY(-1px);
        }

        .interview-id {
            color: var(--text);
            font-size: 0.98rem;
            font-weight: 720;
            letter-spacing: -0.015em;
            overflow-wrap: anywhere;
            line-height: 1.28;
        }

        .interview-source {
            color: var(--text-soft);
            font-size: 0.82rem;
            margin-top: 0.16rem;
            overflow-wrap: anywhere;
        }

        .stat-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.4rem;
            margin-top: 0.85rem;
            margin-bottom: 0.7rem;
        }

        .stat-pill {
            background: var(--surface-muted);
            border: 1px solid var(--border);
            border-radius: 999px;
            color: var(--text-soft);
            font-size: 0.78rem;
            padding: 0.17rem 0.52rem;
        }

        .stat-pill strong {
            color: var(--text);
            font-weight: 720;
        }

        .speaker-chip {
            display: inline-block;
            background: var(--accent-soft);
            color: var(--accent);
            border: 1px solid #dbeafe;
            border-radius: 999px;
            font-size: 0.74rem;
            font-weight: 650;
            padding: 0.15rem 0.52rem;
            margin-right: 0.3rem;
            margin-bottom: 0.25rem;
        }

        .detail-panel {
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: var(--radius-lg);
            padding: 1.15rem;
            margin: 0.85rem 0 1.2rem;
            box-shadow: var(--shadow-sm);
        }

        .detail-title {
            color: var(--text);
            font-size: 1rem;
            font-weight: 720;
            letter-spacing: -0.015em;
            overflow-wrap: anywhere;
        }

        .detail-subtitle {
            color: var(--text-soft);
            font-size: 0.82rem;
            margin-top: 0.18rem;
            overflow-wrap: anywhere;
        }

        .detail-dialog-id {
            color: var(--text);
            font-size: 1.05rem;
            font-weight: 760;
            letter-spacing: -0.015em;
            overflow-wrap: anywhere;
            line-height: 1.25;
        }

        .detail-dialog-source {
            color: var(--text-soft);
            font-size: 0.84rem;
            margin-top: 0.2rem;
            overflow-wrap: anywhere;
            line-height: 1.35;
        }

        .context-banner {
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: var(--radius-lg);
            box-shadow: var(--shadow-sm);
            padding: 0.95rem 1.1rem;
            display: grid;
            grid-template-columns: 1.8fr 1fr 0.75fr;
            gap: 1rem;
            margin-bottom: 0.85rem;
        }

        .context-label {
            color: var(--text-muted);
            font-size: 0.7rem;
            font-weight: 760;
            letter-spacing: 0.07em;
            text-transform: uppercase;
            margin-bottom: 0.16rem;
        }

        .context-value {
            color: var(--text);
            font-size: 0.9rem;
            font-weight: 650;
            overflow-wrap: anywhere;
        }

        .empty-state {
            background: var(--surface);
            border: 1px dashed var(--border-strong);
            border-radius: var(--radius-lg);
            padding: 2rem 1.5rem;
            text-align: center;
            color: var(--text-soft);
            font-size: 0.9rem;
            line-height: 1.5;
            margin: 0.75rem 0 1rem;
        }

        .empty-state strong {
            color: var(--text);
        }

        .thread-entry {
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: var(--radius-md);
            padding: 0.65rem 0.75rem;
            margin-bottom: 0.45rem;
            box-shadow: var(--shadow-sm);
        }

        .thread-entry.active {
            background: var(--accent-soft);
            border-color: #bfdbfe;
        }

        .thread-case {
            color: var(--accent);
            font-size: 0.7rem;
            font-weight: 760;
            letter-spacing: 0.05em;
            text-transform: uppercase;
            overflow-wrap: anywhere;
        }

        .thread-preview {
            color: var(--text);
            font-size: 0.82rem;
            font-weight: 600;
            margin-top: 0.18rem;
            overflow-wrap: anywhere;
            line-height: 1.35;
        }

        .thread-meta {
            color: var(--text-muted);
            font-size: 0.74rem;
            margin-top: 0.12rem;
        }

        .metric-card {
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: var(--radius-lg);
            padding: 1rem 1.05rem;
            box-shadow: var(--shadow-sm);
            min-height: 6rem;
        }

        .metric-card-title {
            color: var(--text-muted);
            font-size: 0.72rem;
            font-weight: 760;
            letter-spacing: 0.07em;
            text-transform: uppercase;
        }

        .metric-card-value {
            color: var(--text);
            font-size: 1.2rem;
            font-weight: 760;
            margin-top: 0.35rem;
        }

        .thin-divider {
            border: none;
            border-top: 1px solid var(--border);
            margin: 1rem 0;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# SAFE HTML HELPERS
# ============================================================
def esc(value: Any) -> str:
    return html.escape(str(value if value is not None else ""), quote=True)


def normalize_text(value: Any) -> str:
    return " ".join(str(value or "").split())


def case_display_name(interview_id: Any) -> str:
    """
    Returns a professional UI-facing case name while preserving the internal
    backend identifier used by Chroma/RAG.

    Examples:
    - chat_random_0 -> CASE-001
    - chat_random_35 -> CASE-036
    - 35 -> CASE-036
    """
    raw = str(interview_id if interview_id is not None else "").strip()

    match = re.search(r"chat_random_(\d+)$", raw)
    if match:
        return f"CASE-{int(match.group(1)) + 1:03d}"

    if raw.isdigit():
        return f"CASE-{int(raw) + 1:03d}"

    return raw or "CASE-UNKNOWN"


def case_search_blob(interview_id: Any) -> str:
    display = case_display_name(interview_id)
    raw = str(interview_id if interview_id is not None else "")
    return f"{display} {raw}"


def case_internal_caption(interview_id: Any) -> str:
    return f"Internal reference: {interview_id}"


# ============================================================
# CACHED BACKEND
# ============================================================
@st.cache_resource
def bootstrap_backend() -> Dict[str, Any]:
    return init_backend()


@st.cache_data(show_spinner=False)
def load_interviews() -> List[Dict[str, Any]]:
    return list_interviews()


@st.cache_data(show_spinner=False)
def load_preview(interview_id: str) -> Dict[str, Any]:
    return get_interview_preview(interview_id)


@st.cache_data(show_spinner=False)
def load_interview_ai_summary(interview_id: str) -> Dict[str, Any]:
    return ask_case(
        question=SUMMARY_QUESTION,
        interview_id=interview_id,
        history=[],
        thread_id=f"summary_{interview_id}",
    )


@st.cache_data(show_spinner=False)
def load_interview_keywords(interview_id: str) -> Dict[str, Any]:
    return ask_case(
        question=KEYWORDS_QUESTION,
        interview_id=interview_id,
        history=[],
        thread_id=f"keywords_{interview_id}",
    )


# ============================================================
# SESSION STATE
# ============================================================
def ensure_session_state() -> None:
    defaults = {
        "authenticated": False,
        "username": None,
        "role": None,
        "display_name": None,
        "login_error": "",
        "login_role_choice": None,
        "user_store": None,
        "new_therapist_username": "",
        "new_therapist_display_name": "",
        "new_therapist_password": "",
        "case_threads": {},
        "active_thread_by_case": {},
        "thread_counter": 0,
        "selected_case": None,
        "active_view": "Interviews",
        "pending_view": None,
        "last_result": None,
        "debug_mode": False,
        "interview_page": 1,
        "interview_search": "",
        "speaker_filter": "All speakers",
        "min_turns_filter": 0,
        "min_chunks_filter": 0,
        "sort_option": SORT_OPTIONS[0],
        "selected_detail_interview": None,
        "detail_dialog_interview": None,
        "open_detail_dialog": False,
        "filters_signature": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    if st.session_state.user_store is None:
        st.session_state.user_store = {
            username: dict(user) for username, user in DEFAULT_USERS.items()
        }


def apply_pending_view() -> None:
    pending = st.session_state.get("pending_view")
    if pending:
        st.session_state.active_view = pending
        st.session_state.pending_view = None


def set_view(view_name: str) -> None:
    st.session_state.active_view = view_name


# ============================================================
# AUTH / ROLE HELPERS
# ============================================================
def is_authenticated() -> bool:
    return bool(st.session_state.get("authenticated"))


def current_role() -> Optional[str]:
    return st.session_state.get("role")


def is_admin() -> bool:
    return current_role() == "Administrator"


def get_allowed_views() -> List[str]:
    role = current_role()
    return ROLE_NAVIGATION.get(role, [])


def logout() -> None:
    st.session_state.authenticated = False
    st.session_state.username = None
    st.session_state.role = None
    st.session_state.display_name = None
    st.session_state.login_error = ""
    st.session_state.active_view = "Interviews"
    st.session_state.pending_view = None
    st.session_state.open_detail_dialog = False
    st.session_state.detail_dialog_interview = None
    st.session_state.selected_detail_interview = None
    st.session_state.debug_mode = False
    st.rerun()


def _complete_login(username: str, user: Dict[str, Any]) -> None:
    st.session_state.authenticated = True
    st.session_state.username = username
    st.session_state.role = user["role"]
    st.session_state.display_name = user["display_name"]
    st.session_state.login_error = ""
    st.session_state.active_view = "Interviews"
    st.session_state.pending_view = None
    st.session_state.open_detail_dialog = False
    st.session_state.detail_dialog_interview = None
    st.session_state.selected_detail_interview = None
    st.rerun()


def get_users() -> Dict[str, Dict[str, Any]]:
    users = st.session_state.get("user_store")
    if users is None:
        users = {username: dict(user) for username, user in DEFAULT_USERS.items()}
        st.session_state.user_store = users
    return users


def attempt_login(username: str, password: str, expected_role: Optional[str] = None) -> None:
    username = username.strip()
    users = get_users()
    user = users.get(username)

    if not user or user.get("password") != password:
        st.session_state.login_error = "Invalid username or password."
        return

    if user.get("status", "Active") != "Active":
        st.session_state.login_error = "This user is not active."
        return

    if expected_role and user.get("role") != expected_role:
        st.session_state.login_error = f"This account is not a {expected_role} account."
        return

    _complete_login(username, user)


def select_login_role(role: str) -> None:
    st.session_state.login_role_choice = role
    st.session_state.login_error = ""


def add_therapist_user(username: str, display_name: str, password: str) -> None:
    username = normalize_text(username).lower()
    display_name = normalize_text(display_name)

    if not username or not password or not display_name:
        st.warning("Please complete username, display name, and password.")
        return

    if username in get_users():
        st.warning("A user with this username already exists.")
        return

    get_users()[username] = {
        "password": password,
        "role": "Therapist",
        "display_name": display_name,
        "status": "Active",
    }
    st.success(f"Therapist user '{username}' was added.")


def deactivate_therapist_user(username: str) -> None:
    users = get_users()
    user = users.get(username)

    if not user:
        st.warning("Selected user was not found.")
        return

    if user.get("role") != "Therapist":
        st.warning("Only therapist users can be deactivated from this prototype panel.")
        return

    if user.get("status") == "Inactive":
        st.info(f"Therapist user '{username}' is already inactive.")
        return

    user["status"] = "Inactive"
    st.success(f"Therapist user '{username}' was deactivated.")


def delete_therapist_user(username: str) -> None:
    """Backward-compatible alias for older button handlers."""
    deactivate_therapist_user(username)


def require_admin(view_name: str) -> None:
    if view_name in ADMIN_ONLY_VIEWS and not is_admin():
        st.error("You do not have permission to access this page.")
        st.stop()


# ============================================================
# THREAD / CHAT STATE
# ============================================================
def next_thread_id() -> str:
    st.session_state.thread_counter += 1
    return f"thread_{st.session_state.thread_counter:03d}"


def ensure_case_threads(case_id: str) -> None:
    if case_id not in st.session_state.case_threads:
        return
    if not st.session_state.case_threads[case_id]:
        return
    active = st.session_state.active_thread_by_case.get(case_id)
    if active not in st.session_state.case_threads[case_id]:
        first = sorted(st.session_state.case_threads[case_id].keys())[0]
        st.session_state.active_thread_by_case[case_id] = first


def create_new_thread(case_id: str) -> None:
    if not case_id:
        return
    if case_id not in st.session_state.case_threads:
        st.session_state.case_threads[case_id] = {}
    thread_id = next_thread_id()
    st.session_state.case_threads[case_id][thread_id] = []
    st.session_state.active_thread_by_case[case_id] = thread_id
    st.session_state.selected_case = case_id
    st.session_state.selected_detail_interview = case_id
    st.session_state.open_detail_dialog = False
    st.session_state.detail_dialog_interview = None
    st.session_state.pending_view = "Clinical Chat"


def activate_thread(case_id: str, thread_id: str) -> None:
    if case_id not in st.session_state.case_threads:
        return
    if thread_id not in st.session_state.case_threads[case_id]:
        return
    st.session_state.selected_case = case_id
    st.session_state.active_thread_by_case[case_id] = thread_id
    st.session_state.selected_detail_interview = case_id
    st.session_state.open_detail_dialog = False
    st.session_state.detail_dialog_interview = None
    st.session_state.pending_view = "Clinical Chat"


def get_active_thread_id(case_id: Optional[str]) -> Optional[str]:
    if not case_id:
        return None
    if case_id not in st.session_state.case_threads:
        return None
    ensure_case_threads(case_id)
    return st.session_state.active_thread_by_case.get(case_id)


def get_thread_messages(case_id: str, thread_id: str) -> List[Dict[str, Any]]:
    if case_id not in st.session_state.case_threads:
        st.session_state.case_threads[case_id] = {}
    return st.session_state.case_threads[case_id].setdefault(thread_id, [])


def append_message(
    case_id: str,
    thread_id: str,
    role: str,
    content: str,
    payload: Optional[Dict[str, Any]] = None,
) -> None:
    messages = get_thread_messages(case_id, thread_id)
    messages.append(
        {
            "role": role,
            "content": content,
            "payload": payload or {},
            "timestamp": time.time(),
        }
    )


def delete_thread(case_id: str, thread_id: str) -> None:
    if case_id not in st.session_state.case_threads:
        return
    if thread_id not in st.session_state.case_threads[case_id]:
        return

    del st.session_state.case_threads[case_id][thread_id]

    if not st.session_state.case_threads[case_id]:
        del st.session_state.case_threads[case_id]
        st.session_state.active_thread_by_case.pop(case_id, None)
    else:
        remaining = sorted(st.session_state.case_threads[case_id].keys())
        current = st.session_state.active_thread_by_case.get(case_id)
        if current == thread_id or current not in remaining:
            st.session_state.active_thread_by_case[case_id] = remaining[0]

    entries = get_thread_history_entries()
    st.session_state.selected_case = entries[0]["case_id"] if entries else None
    st.session_state.last_result = None


def clear_all_histories() -> None:
    st.session_state.case_threads = {}
    st.session_state.active_thread_by_case = {}
    st.session_state.thread_counter = 0
    st.session_state.selected_case = None
    st.session_state.last_result = None
    st.session_state.pending_view = None
    st.session_state.open_detail_dialog = False
    st.session_state.detail_dialog_interview = None


def build_backend_history(messages: List[Dict[str, Any]]) -> List[Tuple[str, str]]:
    history: List[Tuple[str, str]] = []
    pending_question: Optional[str] = None
    for msg in messages:
        if msg.get("role") == "user":
            pending_question = msg.get("content", "")
        elif msg.get("role") == "assistant" and pending_question is not None:
            raw_answer = msg.get("payload", {}).get("raw_answer", msg.get("content", ""))
            history.append((pending_question, raw_answer))
            pending_question = None
    return history[-6:]


def get_thread_history_entries() -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    for case_id, threads in st.session_state.case_threads.items():
        for thread_id, messages in threads.items():
            first_user = next((m.get("content", "") for m in messages if m.get("role") == "user"), "")
            last_ts = messages[-1].get("timestamp", 0.0) if messages else 0.0
            entries.append(
                {
                    "case_id": case_id,
                    "case_display": case_display_name(case_id),
                    "thread_id": thread_id,
                    "preview": shorten_label(first_user),
                    "num_messages": len(messages),
                    "last_ts": last_ts,
                }
            )
    entries.sort(key=lambda x: (x["last_ts"], x["thread_id"]), reverse=True)
    return entries


def sync_active_selection() -> None:
    entries = get_thread_history_entries()
    if not entries:
        st.session_state.selected_case = None
        return

    selected_case = st.session_state.selected_case
    if selected_case and selected_case in st.session_state.case_threads:
        active_thread = st.session_state.active_thread_by_case.get(selected_case)
        if active_thread in st.session_state.case_threads[selected_case]:
            return

    st.session_state.selected_case = entries[0]["case_id"]
    st.session_state.active_thread_by_case[entries[0]["case_id"]] = entries[0]["thread_id"]


# ============================================================
# DATA HELPERS
# ============================================================
def shorten_label(text: str, max_chars: int = 62) -> str:
    text = normalize_text(text)
    if not text:
        return "Empty chat"
    return text[:max_chars] + ("…" if len(text) > max_chars else "")


def paginate_interviews(
    interviews: List[Dict[str, Any]],
    page: int,
    page_size: int = PAGE_SIZE,
) -> Tuple[List[Dict[str, Any]], int, int, int, int]:
    total = len(interviews)
    if total == 0:
        return [], 1, 1, 0, 0
    total_pages = ceil(total / page_size)
    page = max(1, min(page, total_pages))
    start_idx = (page - 1) * page_size
    end_idx = min(start_idx + page_size, total)
    return interviews[start_idx:end_idx], page, total_pages, start_idx + 1, end_idx


def filter_interviews(
    interviews: List[Dict[str, Any]],
    search_query: str,
    speaker_filter: str,
    min_turns: int,
    min_chunks: int,
    sort_option: str,
) -> List[Dict[str, Any]]:
    query = normalize_text(search_query).lower()
    results: List[Dict[str, Any]] = []

    for item in interviews:
        speakers = [str(s) for s in item.get("speakers", [])]
        searchable_blob = " ".join(
            [
                case_search_blob(item.get("interview_id", "")),
                str(item.get("source", "")),
                " ".join(str(src) for src in item.get("sources", [])),
                " ".join(speakers),
            ]
        ).lower()

        if query and query not in searchable_blob:
            continue
        if speaker_filter != "All speakers" and speaker_filter not in speakers:
            continue
        if int(item.get("num_turns", 0) or 0) < min_turns:
            continue
        if int(item.get("num_chunks", 0) or 0) < min_chunks:
            continue
        results.append(item)

    if sort_option == "Case ID · Z to A":
        results.sort(key=lambda x: case_display_name(x.get("interview_id", "")), reverse=True)
    elif sort_option == "Most turns first":
        results.sort(key=lambda x: int(x.get("num_turns", 0) or 0), reverse=True)
    elif sort_option == "Most chunks first":
        results.sort(key=lambda x: int(x.get("num_chunks", 0) or 0), reverse=True)
    else:
        results.sort(key=lambda x: case_display_name(x.get("interview_id", "")))

    return results


def update_page_on_filter_change(signature: Tuple[Any, ...]) -> None:
    previous = st.session_state.get("filters_signature")
    if previous is None:
        st.session_state.filters_signature = signature
        return
    if previous != signature:
        st.session_state.filters_signature = signature
        st.session_state.interview_page = 1


def reset_filters() -> None:
    st.session_state.interview_search = ""
    st.session_state.speaker_filter = "All speakers"
    st.session_state.min_turns_filter = 0
    st.session_state.min_chunks_filter = 0
    st.session_state.sort_option = SORT_OPTIONS[0]
    st.session_state.interview_page = 1
    st.session_state.selected_detail_interview = None
    st.session_state.open_detail_dialog = False
    st.session_state.detail_dialog_interview = None


def error_payload(message: str) -> Dict[str, Any]:
    return {
        "answer": message,
        "answer_with_refs": message,
        "manual_rule": "",
        "evidence": [],
        "manual_sources": [],
        "retrieved_turns": [],
        "elapsed_sec": 0.0,
    }


# ============================================================
# CHAT LOGIC
# ============================================================
def submit_chat_question(question: str, selected_case: str, thread_id: str) -> None:
    if not question or not question.strip():
        return

    append_message(selected_case, thread_id, "user", question)
    history_for_backend = build_backend_history(get_thread_messages(selected_case, thread_id))

    result = ask_case(
        question=question,
        interview_id=selected_case,
        history=history_for_backend,
        thread_id=thread_id,
    )

    append_message(
        selected_case,
        thread_id,
        "assistant",
        result.get("answer_with_refs", result.get("answer", "")),
        payload=result,
    )
    st.session_state.last_result = result


# ============================================================
# UI COMPONENTS
# ============================================================
def render_login_screen() -> None:
    st.markdown(
        """
        <div style="max-width:760px;margin:4.5rem auto 1.6rem auto;text-align:center;">
            <div style="font-size:1.85rem;font-weight:780;letter-spacing:-0.04em;color:var(--text);">
                Clinical Decision Support
            </div>
            <div style="font-size:0.95rem;color:var(--text-soft);margin-top:0.4rem;line-height:1.45;">
                Choose a role, then sign in with a valid account for that role.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    left_space, role_area, right_space = st.columns([0.45, 2.1, 0.45])

    with role_area:
        admin_col, therapist_col = st.columns(2, gap="large")

        with admin_col:
            with st.container(border=True):
                st.markdown("### Administrator")
                st.caption(
                    "Full access for system supervision, user management, RAG evaluation, and debug diagnostics."
                )
                st.markdown(
                    """
                    - Interviews and Clinical Chat
                    - RAG Evaluation
                    - Admin / Debug
                    - User management prototype
                    - Technical payloads and diagnostics
                    """
                )
                if st.button(
                    "Sign in as Administrator",
                    key="select_admin_login",
                    use_container_width=True,
                    type="primary",
                ):
                    select_login_role("Administrator")
                    st.rerun()

        with therapist_col:
            with st.container(border=True):
                st.markdown("### Therapist")
                st.caption(
                    "Clinical user access for interview exploration and case-specific clinical assistance."
                )
                st.markdown(
                    """
                    - Interviews
                    - Interview details, summaries, and keywords
                    - Clinical Chat from a selected interview
                    - Evidence traceability
                    - No admin/debug access
                    """
                )
                if st.button(
                    "Sign in as Therapist",
                    key="select_therapist_login",
                    use_container_width=True,
                    type="primary",
                ):
                    select_login_role("Therapist")
                    st.rerun()

        selected_role = st.session_state.get("login_role_choice")

        if selected_role:
            st.markdown("<div style='height:0.75rem;'></div>", unsafe_allow_html=True)
            with st.container(border=True):
                st.markdown(f"### {selected_role} sign in")
                st.caption("Enter username and password for the selected role.")

                username = st.text_input(
                    "Username",
                    placeholder="admin" if selected_role == "Administrator" else "therapist, therapist2, or therapist3",
                    key="login_username",
                )

                password = st.text_input(
                    "Password",
                    type="password",
                    placeholder="Enter password",
                    key="login_password",
                )

                if st.session_state.get("login_error"):
                    st.error(st.session_state.login_error)

                submit_col, cancel_col = st.columns([1.4, 1])
                with submit_col:
                    if st.button(
                        f"Continue as {selected_role}",
                        use_container_width=True,
                        type="primary",
                    ):
                        attempt_login(username, password, expected_role=selected_role)

                with cancel_col:
                    if st.button("Change role", use_container_width=True, type="secondary"):
                        st.session_state.login_role_choice = None
                        st.session_state.login_error = ""
                        st.rerun()

                st.caption(
                    "Demo accounts: admin / admin123 · therapist / therapist123 · "
                    "therapist2 / therapist2123 · therapist3 / therapist3123"
                )


def render_app_header() -> None:
    st.markdown(
        """
        <div class="app-header">
            <div class="app-title">Clinical Decision Support</div>
            <div class="app-subtitle">Turn-aware RAG · layered manuals · traceable evidence</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_navigation() -> str:
    apply_pending_view()

    allowed_views = get_allowed_views()

    if not allowed_views:
        st.error("No views are available for this role.")
        st.stop()

    if st.session_state.active_view not in allowed_views:
        st.session_state.active_view = allowed_views[0]

    st.markdown('<div class="nav-shell">', unsafe_allow_html=True)

    cols = st.columns(len(allowed_views))

    for col, view in zip(cols, allowed_views):
        is_active = st.session_state.active_view == view

        with col:
            if st.button(
                view,
                key=f"nav_{view}",
                use_container_width=True,
                type="primary" if is_active else "secondary",
            ):
                set_view(view)
                st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)

    return st.session_state.active_view


def render_interview_card(item: Dict[str, Any]) -> None:
    interview_id = str(item["interview_id"])
    display_id = case_display_name(interview_id)
    source = item.get("source", "unknown")
    speakers = [str(s) for s in item.get("speakers", [])]
    is_selected = str(st.session_state.get("selected_detail_interview", "")) == interview_id

    card_class = "interview-card selected" if is_selected else "interview-card"
    speaker_chips = "".join(f'<span class="speaker-chip">{esc(s)}</span>' for s in speakers)
    stat_row = (
        f'<span class="stat-pill"><strong>{esc(item.get("num_turns", 0))}</strong> turns</span>'
        f'<span class="stat-pill"><strong>{esc(item.get("num_chunks", 0))}</strong> chunks</span>'
        f'<span class="stat-pill"><strong>{len(speakers)}</strong> speakers</span>'
    )

    st.markdown(
        f"""
        <div class="{card_class}">
            <div class="interview-id">{esc(display_id)}</div>
            <div class="interview-source">{esc(source)}</div>
            <div class="stat-row">{stat_row}</div>
            <div>{speaker_chips}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    action_a, action_b = st.columns([1, 1])

    with action_a:
        if st.button(
            "Open Clinical Assistant",
            key=f"quick_open_assistant_{interview_id}",
            use_container_width=True,
            type="primary",
        ):
            st.session_state.open_detail_dialog = False
            st.session_state.detail_dialog_interview = None
            create_new_thread(interview_id)
            st.rerun()

    with action_b:
        if st.button(
            "Hide details" if is_selected else "View details",
            key=f"details_{interview_id}",
            use_container_width=True,
            type="secondary",
        ):
            if is_selected:
                st.session_state.selected_detail_interview = None
                st.session_state.open_detail_dialog = False
                st.session_state.detail_dialog_interview = None
            else:
                st.session_state.selected_detail_interview = interview_id
                st.session_state.detail_dialog_interview = interview_id
                st.session_state.open_detail_dialog = True

            st.rerun()


@st.dialog("Interview details", width="large")
def render_interview_details_dialog(interview_id: str) -> None:
    display_id = case_display_name(interview_id)
    preview = load_preview(str(interview_id))

    if not preview.get("found"):
        st.warning("No preview data available for this interview.")

        if st.button("Close", use_container_width=True, type="secondary"):
            st.session_state.selected_detail_interview = None
            st.session_state.open_detail_dialog = False
            st.session_state.detail_dialog_interview = None
            st.rerun()

        return

    st.markdown(
        f"""
        <div class="section-label">Selected case</div>
        <div class="detail-dialog-id">{esc(display_id)}</div>
        <div class="detail-dialog-source">{esc(preview.get("source", "unknown"))}</div>
        """,
        unsafe_allow_html=True,
    )

    if is_admin():
        st.caption(case_internal_caption(interview_id))

    st.markdown('<hr class="thin-divider">', unsafe_allow_html=True)

    st.markdown("#### Metadata")
    m1, m2, m3 = st.columns(3)
    m1.metric("Turns", preview.get("num_turns", 0))
    m2.metric("Speakers", len(preview.get("speakers", [])))
    m3.metric("Sources", len(preview.get("sources", [])))

    speakers = preview.get("speakers", [])
    if speakers:
        st.caption("Speakers: " + ", ".join(str(s) for s in speakers))

    sources = preview.get("sources", [])
    if sources:
        with st.expander("Source files", expanded=False):
            for source in sources:
                st.write(str(source))

    st.markdown('<hr class="thin-divider">', unsafe_allow_html=True)

    st.markdown("#### AI-generated summary")
    st.caption(
        "Generated from the selected interview using the RAG backend. "
        "The summary is constrained to explicit interview content."
    )

    try:
        with st.spinner("Generating interview summary..."):
            summary_result = load_interview_ai_summary(str(interview_id))

        summary_text = summary_result.get(
            "answer_with_refs",
            summary_result.get("answer", ""),
        )

        if summary_text:
            st.markdown(summary_text)
        else:
            st.caption("No summary could be generated for this interview.")

        summary_evidence = summary_result.get("evidence", [])
        if summary_evidence:
            with st.expander("Summary evidence", expanded=False):
                render_evidence_items(summary_evidence)

    except Exception as exc:
        st.warning(f"Summary could not be generated: {exc}")

    st.markdown('<hr class="thin-divider">', unsafe_allow_html=True)

    st.markdown("#### Keywords and clinical signals")
    st.caption("Explicit keywords extracted from the selected interview.")

    try:
        with st.spinner("Extracting keywords..."):
            keywords_result = load_interview_keywords(str(interview_id))

        keywords_text = keywords_result.get(
            "answer_with_refs",
            keywords_result.get("answer", ""),
        )

        if keywords_text:
            st.markdown(keywords_text)
        else:
            st.caption("No keywords could be extracted for this interview.")

    except Exception as exc:
        st.warning(f"Keywords could not be extracted: {exc}")

    st.markdown('<hr class="thin-divider">', unsafe_allow_html=True)

    st.markdown("#### Conversation excerpt")
    preview_turns = preview.get("preview_turns", [])

    if not preview_turns:
        st.caption("No conversation turns available.")
    else:
        for turn in preview_turns:
            label = f"Turn {turn.get('turn_id')} · {turn.get('speaker', 'unknown')}"
            if turn.get("lines"):
                label += f" · lines {turn.get('lines')}"
            with st.expander(label, expanded=False):
                st.write(turn.get("text", ""))

    st.markdown('<hr class="thin-divider">', unsafe_allow_html=True)

    cta_col, close_col = st.columns([1.4, 1])

    with cta_col:
        if st.button(
            "Open Clinical Assistant",
            key=f"dialog_open_assistant_{interview_id}",
            use_container_width=True,
            type="primary",
        ):
            st.session_state.open_detail_dialog = False
            st.session_state.detail_dialog_interview = None
            create_new_thread(str(interview_id))
            st.rerun()

    with close_col:
        if st.button(
            "Close",
            key=f"dialog_close_{interview_id}",
            use_container_width=True,
            type="secondary",
        ):
            st.session_state.selected_detail_interview = None
            st.session_state.open_detail_dialog = False
            st.session_state.detail_dialog_interview = None
            st.rerun()


def render_context_banner(case_id: str, thread_id: str, message_count: int) -> None:
    display_id = case_display_name(case_id)
    st.markdown(
        f"""
        <div class="context-banner">
            <div>
                <div class="context-label">Case</div>
                <div class="context-value">{esc(display_id)}</div>
            </div>
            <div>
                <div class="context-label">Session</div>
                <div class="context-value">{esc(thread_id)}</div>
            </div>
            <div>
                <div class="context-label">Messages</div>
                <div class="context-value">{esc(message_count)}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_related_interview_panel(case_id: str) -> None:
    if not case_id:
        return
    with st.expander("View related interview excerpt", expanded=False):
        preview = load_preview(case_id)
        if not preview.get("found"):
            st.caption("No interview preview available.")
            return
        st.caption(
            f"Case: {case_display_name(case_id)} · "
            f"Source: {preview.get('source', 'unknown')} · "
            f"Turns: {preview.get('num_turns', 0)} · "
            f"Speakers: {', '.join(str(s) for s in preview.get('speakers', []))}"
        )
        for turn in preview.get("preview_turns", []):
            label = f"Turn {turn.get('turn_id')} · {turn.get('speaker', 'unknown')}"
            if turn.get("lines"):
                label += f" · lines {turn.get('lines')}"
            with st.expander(label, expanded=False):
                st.write(turn.get("text", ""))


def render_evidence_items(evidence_items: List[Dict[str, Any]]) -> None:
    st.markdown("**Evidence traceability**")
    if not evidence_items:
        st.caption("No traceable quotes detected.")
        return

    for item in evidence_items:
        label = f"[{item.get('ref')}] {item.get('source', 'unknown')} · {item.get('speaker', 'unknown')}"
        if item.get("turn_id") is not None:
            label += f" · turn {item.get('turn_id')}"
        if item.get("lines"):
            label += f" · lines {item.get('lines')}"
        with st.expander(label, expanded=False):
            st.markdown("**Exact quote**")
            st.code(item.get("quote", ""))


def render_manual_sources(manual_sources: List[Dict[str, Any]]) -> None:
    if not manual_sources:
        return
    with st.expander("Manual sources used", expanded=False):
        for item in manual_sources:
            st.markdown(
                f"- **{item.get('label', '')}** {item.get('source', 'unknown')} "
                f"(layer={item.get('layer_num', '')})"
            )
            excerpt = item.get("excerpt", "")
            if excerpt:
                st.caption(excerpt)


def render_retrieved_turns(turns: List[Dict[str, Any]]) -> None:
    if not turns:
        return
    with st.expander("Retrieved interview turns", expanded=False):
        for turn in turns:
            title = f"Turn {turn.get('turn_id')} · {turn.get('speaker', 'unknown')}"
            if turn.get("lines"):
                title += f" · lines {turn.get('lines')}"
            st.markdown(f"**{title}**")
            st.write(turn.get("text", ""))
            st.markdown("---")


def render_assistant_payload(payload: Dict[str, Any]) -> None:
    st.markdown(payload.get("answer_with_refs", payload.get("answer", "(empty)")))

    manual_rule = payload.get("manual_rule", "")
    if manual_rule and is_admin():
        st.caption(f"Manual rule: {manual_rule}")

    render_evidence_items(payload.get("evidence", []))
    render_manual_sources(payload.get("manual_sources", []))
    render_retrieved_turns(payload.get("retrieved_turns", []))

    elapsed = payload.get("elapsed_sec")
    if elapsed is not None:
        st.caption(f"Latency: {elapsed:.2f} s")


def render_chat_messages(messages: List[Dict[str, Any]]) -> None:
    for msg in messages:
        with st.chat_message(msg.get("role", "assistant")):
            if msg.get("role") == "assistant":
                render_assistant_payload(msg.get("payload", {}))
            else:
                st.markdown(msg.get("content", ""))


def render_thread_history_sidebar(selected_case: Optional[str], active_thread: Optional[str]) -> None:
    entries = get_thread_history_entries()
    st.markdown('<div class="section-label">Chat history</div>', unsafe_allow_html=True)

    if not entries:
        st.caption("No chat history yet. Start a session from an interview card.")
        return

    for entry in entries:
        is_active = entry["case_id"] == selected_case and entry["thread_id"] == active_thread
        css_class = "thread-entry active" if is_active else "thread-entry"
        st.markdown(
            f"""
            <div class="{css_class}">
                <div class="thread-case">{esc(entry.get("case_display", case_display_name(entry["case_id"])))}</div>
                <div class="thread-preview">{esc(entry["preview"])}</div>
                <div class="thread-meta">{esc(entry["num_messages"])} messages · {esc(entry["thread_id"])}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        open_col, delete_col = st.columns([5, 1])
        with open_col:
            if st.button(
                "Current chat" if is_active else "Open chat",
                key=f"open_thread_{entry['case_id']}_{entry['thread_id']}",
                use_container_width=True,
                disabled=is_active,
                type="secondary",
            ):
                activate_thread(entry["case_id"], entry["thread_id"])
                st.rerun()
        with delete_col:
            if st.button(
                "×",
                key=f"delete_thread_{entry['case_id']}_{entry['thread_id']}",
                use_container_width=True,
                type="secondary",
            ):
                delete_thread(entry["case_id"], entry["thread_id"])
                st.rerun()


def render_metric_card(title: str, value: Any, body: str = "") -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-card-title">{esc(title)}</div>
            <div class="metric-card-value">{esc(value)}</div>
            <div class="muted-copy">{esc(body)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# BOOTSTRAP
# ============================================================
inject_custom_css()
ensure_session_state()

if not is_authenticated():
    render_login_screen()
    st.stop()

try:
    backend_info = bootstrap_backend()
except Exception as exc:
    st.error(f"Backend initialization error: {exc}")
    st.stop()

interviews = load_interviews()
if not interviews:
    st.error("No interviews were found in the indexed collection.")
    st.stop()

sync_active_selection()
selected_case = st.session_state.selected_case
active_thread = get_active_thread_id(selected_case)
current_messages = get_thread_messages(selected_case, active_thread) if selected_case and active_thread else []


# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown(
        f"""
        <div style="margin-bottom:1rem;">
            <div style="font-size:0.72rem;font-weight:760;letter-spacing:0.08em;
                        text-transform:uppercase;color:var(--text-muted);">
                Signed in as
            </div>
            <div style="font-size:0.98rem;font-weight:720;color:var(--text);margin-top:0.15rem;">
                {esc(st.session_state.display_name)}
            </div>
            <div style="font-size:0.82rem;color:var(--text-soft);margin-top:0.1rem;">
                Role: {esc(st.session_state.role)}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<hr class="thin-divider">', unsafe_allow_html=True)

    st.markdown(
        """
        <div style="margin-bottom:0.95rem;">
            <div style="font-size:1rem;font-weight:760;color:var(--text);letter-spacing:-0.02em;">Clinical chats</div>
            <div style="font-size:0.82rem;color:var(--text-soft);margin-top:0.15rem;line-height:1.35;">
                Sessions can only be created from an interview.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.button("Clear chat history", use_container_width=True, type="secondary"):
        clear_all_histories()
        st.rerun()

    st.markdown('<hr class="thin-divider">', unsafe_allow_html=True)

    selected_case = st.session_state.selected_case
    active_thread = get_active_thread_id(selected_case)
    render_thread_history_sidebar(selected_case, active_thread)

sync_active_selection()
selected_case = st.session_state.selected_case
active_thread = get_active_thread_id(selected_case)
current_messages = get_thread_messages(selected_case, active_thread) if selected_case and active_thread else []


# ============================================================
# MAIN HEADER + NAVIGATION
# ============================================================
render_app_header()
selected_view = render_navigation()
require_admin(selected_view)


# ============================================================
# VIEW: INTERVIEWS
# ============================================================
if selected_view == "Interviews":
    st.markdown("### Case list")
    st.caption("Browse indexed clinical cases, inspect details, and start a case-specific clinical chat.")

    all_speakers = sorted({str(s) for item in interviews for s in item.get("speakers", []) if s})
    speaker_options = ["All speakers"] + all_speakers
    if st.session_state.speaker_filter not in speaker_options:
        st.session_state.speaker_filter = "All speakers"

    with st.container(border=True):
        st.markdown("**Search and filters**")
        f1, f2, f3, f4, f5 = st.columns([2.4, 1.35, 0.9, 0.9, 1.35])
        with f1:
            st.text_input(
                "Search",
                key="interview_search",
                placeholder="Case ID, internal reference, source, or speaker...",
            )
        with f2:
            st.selectbox(
                "Speaker",
                speaker_options,
                key="speaker_filter",
            )
        with f3:
            st.number_input(
                "Min turns",
                min_value=0,
                max_value=500,
                step=1,
                key="min_turns_filter",
            )
        with f4:
            st.number_input(
                "Min chunks",
                min_value=0,
                max_value=5000,
                step=1,
                key="min_chunks_filter",
            )
        with f5:
            st.selectbox(
                "Sort by",
                SORT_OPTIONS,
                key="sort_option",
            )

        if st.button("Reset filters", use_container_width=False, type="secondary"):
            reset_filters()
            st.rerun()

    filter_signature = (
        st.session_state.interview_search,
        st.session_state.speaker_filter,
        int(st.session_state.min_turns_filter),
        int(st.session_state.min_chunks_filter),
        st.session_state.sort_option,
    )
    update_page_on_filter_change(filter_signature)

    filtered_interviews = filter_interviews(
        interviews=interviews,
        search_query=st.session_state.interview_search,
        speaker_filter=st.session_state.speaker_filter,
        min_turns=int(st.session_state.min_turns_filter),
        min_chunks=int(st.session_state.min_chunks_filter),
        sort_option=st.session_state.sort_option,
    )

    paged_interviews, current_page, total_pages, start_item, end_item = paginate_interviews(
        filtered_interviews,
        st.session_state.interview_page,
        PAGE_SIZE,
    )
    st.session_state.interview_page = current_page

    if not filtered_interviews:
        st.markdown(
            """
            <div class="empty-state">
                <strong>No interviews match the current filters.</strong><br>
                Try a broader search query or reset the filters.
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        summary_text = f"Showing {start_item}-{end_item} of {len(filtered_interviews)} cases"
        if len(filtered_interviews) < len(interviews):
            summary_text += f" · filtered from {len(interviews)} total"
        st.caption(summary_text)

        columns = st.columns(2)
        for idx, item in enumerate(paged_interviews):
            with columns[idx % 2]:
                render_interview_card(item)

        st.markdown('<hr class="thin-divider">', unsafe_allow_html=True)
        nav_prev, nav_info, nav_next = st.columns([1, 2, 1])
        with nav_prev:
            if st.button(
                "← Previous",
                disabled=current_page <= 1,
                use_container_width=True,
                type="secondary",
            ):
                st.session_state.interview_page = current_page - 1
                st.rerun()
        with nav_info:
            st.markdown(
                f"<div style='text-align:center;color:var(--text-soft);font-size:0.88rem;padding-top:0.55rem;'>Page {current_page} of {total_pages}</div>",
                unsafe_allow_html=True,
            )
        with nav_next:
            if st.button(
                "Next →",
                disabled=current_page >= total_pages,
                use_container_width=True,
                type="secondary",
            ):
                st.session_state.interview_page = current_page + 1
                st.rerun()

    if (
        st.session_state.get("open_detail_dialog")
        and st.session_state.get("detail_dialog_interview")
    ):
        render_interview_details_dialog(st.session_state.detail_dialog_interview)


# ============================================================
# VIEW: CLINICAL CHAT
# ============================================================
elif selected_view == "Clinical Chat":
    st.markdown("### Clinical RAG chat")

    if not selected_case or not active_thread:
        st.markdown(
            """
            <div class="empty-state">
                <strong>No active clinical session.</strong><br>
                Go to <em>Interviews</em> and open the <strong>Clinical Assistant</strong> from any case card.
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        render_context_banner(selected_case, active_thread, len(current_messages))
        st.caption("Responses are grounded only in the selected interview and include traceable evidence.")
        render_related_interview_panel(selected_case)

        st.markdown("**Suggested clinical questions**")

        def handle_suggestion(question: str, idx: int) -> None:
            if st.button(question, key=f"suggestion_{idx}", use_container_width=True, type="secondary"):
                try:
                    submit_chat_question(question, selected_case, active_thread)
                    st.rerun()
                except Exception as exc:
                    message = f"Error: {exc}"
                    append_message(selected_case, active_thread, "assistant", message, error_payload(message))
                    st.session_state.last_result = None
                    st.rerun()

        pair_end = len(EXAMPLE_QUESTIONS) - 1 if len(EXAMPLE_QUESTIONS) % 2 else len(EXAMPLE_QUESTIONS)
        for idx in range(0, pair_end, 2):
            q_col_a, q_col_b = st.columns(2)
            with q_col_a:
                handle_suggestion(EXAMPLE_QUESTIONS[idx], idx)
            with q_col_b:
                handle_suggestion(EXAMPLE_QUESTIONS[idx + 1], idx + 1)
        if len(EXAMPLE_QUESTIONS) % 2:
            handle_suggestion(EXAMPLE_QUESTIONS[-1], len(EXAMPLE_QUESTIONS) - 1)

        st.markdown('<hr class="thin-divider">', unsafe_allow_html=True)

        if not current_messages:
            st.caption("No messages yet. Use a suggested question above or type your own question below.")

        render_chat_messages(current_messages)

        prompt = st.chat_input("Ask a clinical question about this interview...")
        if prompt:
            try:
                submit_chat_question(prompt, selected_case, active_thread)
                st.rerun()
            except Exception as exc:
                message = f"Error: {exc}"
                append_message(selected_case, active_thread, "assistant", message, error_payload(message))
                st.session_state.last_result = None
                st.rerun()


# ============================================================
# VIEW: RAG EVALUATION
# ============================================================
elif selected_view == "RAG Evaluation":
    require_admin(selected_view)

    st.markdown("### RAG Evaluation")
    st.caption(
        "Basic evaluation over a small labelled dataset: answer generation, retrieval, "
        "traceability, latency, overlap with ground truth, and risk-question alignment."
    )

    output_dir = Path("basic_eval_outputs")
    rows_path = output_dir / "basic_eval_rows.csv"
    summary_path = output_dir / "basic_eval_summary.csv"
    metadata_path = output_dir / "basic_eval_metadata.json"

    if not rows_path.exists() or not summary_path.exists():
        st.markdown(
            """
            <div class="empty-state">
                <strong>No basic evaluation results found.</strong><br>
                Run the basic evaluation script and reload this page.
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("**Recommended commands**")
        st.code(
            """python run_basic_eval.py --input ground_truth_long.csv --limit 40
python run_basic_eval.py --input ground_truth_long.csv --sample-per-question 5
python run_basic_eval.py --input rag_eval_dataset.csv --from-existing --output-dir basic_eval_outputs_existing""",
            language="bash",
        )

        st.markdown("**Expected output files**")
        st.markdown(
            """
            - `basic_eval_outputs/basic_eval_rows.csv`
            - `basic_eval_outputs/basic_eval_summary.csv`
            - `basic_eval_outputs/basic_eval_metadata.json`
            """
        )

    else:
        rows_df = pd.read_csv(rows_path)
        if "interview_id" in rows_df.columns and "case_id" not in rows_df.columns:
            rows_df.insert(0, "case_id", rows_df["interview_id"].apply(case_display_name))
        summary_df = pd.read_csv(summary_path)

        summary_lookup = {
            str(row["metric"]): row["value"]
            for _, row in summary_df.iterrows()
        }

        def metric_value(metric_name: str, default: Any = 0) -> Any:
            value = summary_lookup.get(metric_name, default)
            if pd.isna(value):
                return default
            return value

        st.markdown("#### Evaluation overview")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Examples", int(metric_value("total_examples", 0)))
        c2.metric("Answers", f"{float(metric_value('answer_generation_rate_pct', 0)):.1f}%")
        c3.metric("Contexts", f"{float(metric_value('context_retrieval_rate_pct', 0)):.1f}%")

        risk_value = summary_lookup.get("risk_alignment_rate_pct")
        c4.metric(
            "Risk alignment",
            "N/A" if risk_value is None or pd.isna(risk_value) else f"{float(risk_value):.1f}%",
        )

        c5, c6, c7, c8 = st.columns(4)
        c5.metric("Evidence items", f"{float(metric_value('evidence_item_rate_pct', 0)):.1f}%")
        c6.metric("Format OK", f"{float(metric_value('format_ok_rate_pct', 0)):.1f}%")
        c7.metric("Avg. turns", f"{float(metric_value('avg_retrieved_turns', 0)):.2f}")

        latency = summary_lookup.get("avg_latency_sec")
        c8.metric(
            "Avg. latency",
            "N/A" if latency is None or pd.isna(latency) else f"{float(latency):.2f}s",
        )

        st.info(
            "Interpretation note: context and evidence percentages can be lower for "
            "absence-oriented answers such as 'not explicitly mentioned'. In those cases, "
            "the system may correctly answer that something is not present in the interview "
            "without attaching quote-level evidence. These metrics should therefore be read "
            "together with the per-question table."
        )

        st.markdown("#### Metric summary")
        st.caption("Aggregated metrics generated by `run_basic_eval.py`.")
        st.dataframe(summary_df, use_container_width=True, hide_index=True)

        st.markdown("#### Per-question results")
        st.caption(
            "Each row corresponds to one labelled evaluation question. "
            "Use this table to inspect retrieval, evidence, overlap and risk alignment."
        )

        compact_cols = [
            "case_id",
            "question_id",
            "answer_generated",
            "has_contexts",
            "num_retrieved_turns",
            "has_evidence_items",
            "ground_truth_overlap",
            "is_risk_question",
            "risk_alignment",
            "latency_sec",
            "error",
        ]

        full_cols = [
            "case_id",
            "interview_id",
            "question_id",
            "question",
            "ground_truth",
            "answer",
            "answer_generated",
            "has_contexts",
            "num_contexts",
            "num_retrieved_turns",
            "has_evidence_items",
            "num_evidence_items",
            "ground_truth_overlap",
            "is_risk_question",
            "risk_alignment",
            "latency_sec",
            "error",
        ]

        table_control_col, filter_col_a, filter_col_b, filter_col_c = st.columns([1.2, 1, 1, 1])

        with table_control_col:
            table_view = st.radio(
                "Table view",
                ["Compact", "Full"],
                horizontal=True,
                index=0,
                key="rag_eval_table_view",
            )

        with filter_col_a:
            show_only_errors = st.checkbox("Show only errors", value=False)
        with filter_col_b:
            show_only_risk = st.checkbox("Show only risk questions", value=False)
        with filter_col_c:
            show_only_no_context = st.checkbox("Show only rows without contexts", value=False)

        display_df = rows_df.copy()

        if show_only_errors and "error" in display_df.columns:
            display_df = display_df[
                display_df["error"].fillna("").astype(str).str.strip() != ""
            ]

        if show_only_risk and "is_risk_question" in display_df.columns:
            display_df = display_df[
                display_df["is_risk_question"].fillna(False).astype(bool)
            ]

        if show_only_no_context and "has_contexts" in display_df.columns:
            display_df = display_df[
                ~display_df["has_contexts"].fillna(False).astype(bool)
            ]

        selected_cols = compact_cols if table_view == "Compact" else full_cols
        visible_cols = [col for col in selected_cols if col in display_df.columns]

        st.dataframe(display_df[visible_cols], use_container_width=True, hide_index=True)

        if table_view == "Compact":
            with st.expander("Show text fields for selected rows", expanded=False):
                text_cols = [
                    "case_id",
                    "interview_id",
                    "question_id",
                    "question",
                    "ground_truth",
                    "answer",
                ]
                text_cols = [col for col in text_cols if col in display_df.columns]
                st.dataframe(display_df[text_cols], use_container_width=True, hide_index=True)

        download_col_a, download_col_b = st.columns(2)

        with download_col_a:
            with open(rows_path, "rb") as f:
                st.download_button(
                    "Download per-question CSV",
                    data=f,
                    file_name="basic_eval_rows.csv",
                    mime="text/csv",
                    use_container_width=True,
                    type="secondary",
                )

        with download_col_b:
            with open(summary_path, "rb") as f:
                st.download_button(
                    "Download summary CSV",
                    data=f,
                    file_name="basic_eval_summary.csv",
                    mime="text/csv",
                    use_container_width=True,
                    type="secondary",
                )

        st.markdown("#### How to interpret these metrics")
        st.markdown(
            """
            - **Answers**: percentage of evaluation questions with a generated answer.
            - **Contexts**: percentage of questions where at least one context/turn was retrieved.
            - **Evidence items**: percentage of answers with traceable quote-level evidence.
            - **Format OK**: percentage of backend outputs following the required `Response / Evidence / Manual rule` format.
            - **Risk alignment**: for risk-related questions, whether the answer polarity matches the ground truth.
            """
        )

# ============================================================
# VIEW: ADMIN / DEBUG
# ============================================================
elif selected_view == "Admin / Debug":
    require_admin(selected_view)

    st.markdown("### Admin and Debug")
    st.caption("Administrator-only area for user management, evaluation support, and system diagnostics.")

    st.markdown("#### User management")
    st.caption(
        "User management is session-based in this prototype. "
        "New users and status changes are not persisted after restarting the app."
    )

    user_rows = [
        {
            "Username": username,
            "Display name": data["display_name"],
            "Role": data["role"],
            "Status": data.get("status", "Active"),
        }
        for username, data in get_users().items()
    ]

    st.dataframe(user_rows, use_container_width=True, hide_index=True)

    add_col, deactivate_col = st.columns(2)

    with add_col:
        with st.container(border=True):
            st.markdown("**Add therapist**")
            st.text_input(
                "Username",
                key="new_therapist_username",
                placeholder="e.g. therapist4",
            )
            st.text_input(
                "Display name",
                key="new_therapist_display_name",
                placeholder="e.g. Therapist Four",
            )
            st.text_input(
                "Temporary password",
                key="new_therapist_password",
                type="password",
                placeholder="Set a demo password",
            )
            if st.button("Add therapist", use_container_width=True, type="primary"):
                add_therapist_user(
                    st.session_state.new_therapist_username,
                    st.session_state.new_therapist_display_name,
                    st.session_state.new_therapist_password,
                )
                st.rerun()

    with deactivate_col:
        with st.container(border=True):
            st.markdown("**Deactivate therapist**")
            therapist_users = [
                username
                for username, data in get_users().items()
                if data.get("role") == "Therapist" and data.get("status", "Active") == "Active"
            ]
            if therapist_users:
                selected_therapist = st.selectbox(
                    "Therapist account",
                    therapist_users,
                    key="deactivate_therapist_select",
                )
                st.caption("Only therapist accounts can be deactivated from this prototype panel.")
                if st.button("Deactivate selected therapist", use_container_width=True, type="secondary"):
                    deactivate_therapist_user(selected_therapist)
                    st.rerun()
            else:
                st.caption("No therapist accounts are available to deactivate.")

    with st.expander("Prototype notes", expanded=False):
        st.markdown(
            """
            This prototype stores users in Streamlit session state for demonstration purposes.
            In a production deployment, authentication should be replaced by a secure
            user database, encrypted passwords, role-based access control, and audit logs.
            """
        )

    st.markdown('<hr class="thin-divider">', unsafe_allow_html=True)

    st.markdown("#### System diagnostics")
    st.caption("Technical runtime information for administrator review.")

    st.session_state.debug_mode = st.toggle("Debug mode", value=st.session_state.debug_mode)

    cfg_col, session_col = st.columns(2)
    with cfg_col:
        with st.container(border=True):
            st.markdown("**Active configuration**")
            st.write(f"Model: `{HF_MODEL}`")
            st.write(f"Selected case: `{case_display_name(selected_case) if selected_case else None}`")
            st.write(f"Active chat session: `{active_thread}`")
            st.write(f"Debug mode: `{st.session_state.debug_mode}`")

    with session_col:
        with st.container(border=True):
            st.markdown("**Session summary**")
            st.write(f"Cases with chat history: `{len(st.session_state.case_threads)}`")
            total_threads = sum(len(v) for v in st.session_state.case_threads.values())
            st.write(f"Chat sessions: `{total_threads}`")
            st.write(f"Messages in active session: `{len(current_messages)}`")
            st.write(f"Vector collection: `{backend_info.get('vectorstore_collection', 'unknown')}`")

    if st.session_state.last_result:
        result = st.session_state.last_result
        st.markdown("**Last generated result**")
        m1, m2, m3 = st.columns(3)
        m1.metric("Retrieved turns", len(result.get("retrieved_turns", [])))
        m2.metric("Evidence refs", len(result.get("evidence", [])))
        m3.metric("Latency", f"{result.get('elapsed_sec', 0.0):.2f} s")

        st.markdown("**Raw answer**")
        st.code(result.get("raw_answer", ""))

        st.markdown("**Evidence text**")
        st.code(result.get("evidence_text", ""))

        st.markdown("**Manual rule**")
        st.write(result.get("manual_rule", ""))

        if st.session_state.debug_mode:
            st.markdown("**Full payload**")
            st.json(result)
    else:
        st.caption("No results generated in this session yet.")


# ============================================================
# FOOTER
# ============================================================
st.markdown('<hr class="thin-divider">', unsafe_allow_html=True)
footer_left, footer_right = st.columns([5, 1])
with footer_left:
    st.caption("Clinical Decision Support System · case exploration, clinical chat, and evidence traceability.")
with footer_right:
    if st.button("Logout", key="footer_logout", use_container_width=True, type="secondary"):
        logout()
