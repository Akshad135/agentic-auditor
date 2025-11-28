import streamlit as st
import os
import json
import sys
from pathlib import Path
from dotenv import load_dotenv
from qdrant_client import QdrantClient

load_dotenv()

from src.ingestion.pdf_parser import parse_pdf
from src.agents.workflow import build_agent_graph
from src.config import DATA_DIR, REPORT_DIR, VECTOR_DB_PATH, COLLECTION_NAME, DERIVED_PLAYBOOK_JSON, RAW_DATA_CSV

sys.path.append(str(Path(__file__).resolve().parent))
from development.derive_playbook import derive_policies
from development.download_data import download_legal_data
from development.setup_db import initialize_memory

TEMP_DIR = DATA_DIR / "temp_uploads"
os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

st.set_page_config(page_title="Legal Audit Dashboard", layout="wide")

st.markdown(
    """
    <style>
    .stTextArea textarea { font-family: monospace; }
    .risk-box { 
        padding: 1rem; 
        border-left: 5px solid #ff4b4b; 
        background-color: #262730;
        color: #ffffff;
        margin-bottom: 1rem;
        border-radius: 4px;
    }
    .safe-box { 
        padding: 1rem; 
        border-left: 5px solid #4caf50; 
        background-color: #262730;
        color: #ffffff;
        margin-bottom: 1rem;
        border-radius: 4px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Legal Audit Dashboard")
st.markdown("---")


def ensure_system_health():
    """Check if vector DB exists. If not, auto-repair it with UI feedback."""
    db_ready = False

    if VECTOR_DB_PATH.exists():
        try:
            client = QdrantClient(path=str(VECTOR_DB_PATH))
            collections = [c.name for c in client.get_collections().collections]
            if COLLECTION_NAME in collections:
                count = client.count(COLLECTION_NAME).count
                if count > 0:
                    db_ready = True
            client.close()
        except Exception:
            pass

    if db_ready:
        return True

    # Not ready -> run repair sequence
    with st.status("SYSTEM: initializing (first run)", expanded=True) as status:
        if not DERIVED_PLAYBOOK_JSON.exists():
            st.write("SYSTEM: playbook missing")
            if not RAW_DATA_CSV.exists():
                st.write("SYSTEM: downloading raw legal dataset")
                download_legal_data()

            st.write("SYSTEM: deriving policies (may take a moment)")
            derive_policies()

        st.write("SYSTEM: indexing policies into vector DB")
        initialize_memory()

        status.update(label="SYSTEM: repaired and ready", state="complete", expanded=False)

    return True


with st.sidebar:
    st.header("System Status")

    if ensure_system_health():
        st.success("Vector DB: Active")
    else:
        st.error("System: Critical Error")

    st.markdown("---")
    st.caption("Powered by Agentic Auditor")


input_mode = st.radio("Input Method", ["Upload Document (PDF)", "Paste Text"], horizontal=True)

chunks = []
source_name = ""

if input_mode == "Upload Document (PDF)":
    uploaded_file = st.file_uploader("Select PDF Contract", type="pdf")
    if uploaded_file:
        source_name = uploaded_file.name
        temp_path = TEMP_DIR / "current_audit.pdf"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        with st.status("INGEST: parsing document", expanded=False) as status:
            st.write("INGEST: parsing PDF structure")
            chunks = parse_pdf(str(temp_path))
            status.update(label="INGEST: document ready", state="complete", expanded=False)

        st.info(f"Loaded {len(chunks)} segments from {source_name}")

else:
    raw_text = st.text_area("Paste Clause / Contract Text", height=300)
    if raw_text:
        source_name = "Manual Input"
        chunks = [raw_text] if len(raw_text) > 10 else []


if st.button("Start Audit", type="primary", disabled=not chunks):

    report_data = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    app = build_agent_graph()
    log_container = st.container()

    total = len(chunks)
    for i, chunk in enumerate(chunks):
        progress = (i + 1) / total
        progress_bar.progress(progress)
        status_text.text(f"[AUDIT] Processing segment {i + 1} of {total}...")

        input_state = {
            "section_text": chunk,
            "iteration_count": 0,
        }

        try:
            result = app.invoke(input_state)

            assessment = result.get("risk_assessment", "No assessment provided.")
            risk_found = result.get("risk_found", False)

            # Safety-net check (text-based)
            if not risk_found:
                upper_text = assessment.upper()
                danger_keywords = ["HIGH RISK", "POLICY VIOLATION"]
                safe_keywords = ["NO VIOLATION", "COMPLIANT"]
                if any(d in upper_text for d in danger_keywords) and not any(s in upper_text for s in safe_keywords):
                    risk_found = True

            if risk_found:
                report_item = {
                    "chunk_id": i + 1,
                    "text_preview": (chunk[:200] + "...") if len(chunk) > 200 else chunk,
                    "assessment": assessment,
                    "iterations": result.get("iteration_count", 0),
                }
                report_data.append(report_item)

                # Use a clear separator and label for each risk entry
                with log_container:
                    st.markdown(
                        f"""
                        <hr>
                        <div class="risk-box">
                            <strong>--- RISK DETECTED (Segment {i + 1}) ---</strong><br>
                            <pre style="white-space:pre-wrap; font-family:monospace;">{assessment}</pre>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
            else:
                with log_container:
                    st.markdown(f"<div class='safe-box'>--- SAFE (Segment {i+1}) ---</div>", unsafe_allow_html=True)

        except Exception as e:
            st.error(f"[AUDIT] Error processing segment {i + 1}: {e}")

    # Final report
    st.markdown("---")
    st.header("Audit Report")

    if report_data:
        st.error(f"Audit complete: {len(report_data)} risks identified")

        safe_filename = source_name.replace(" ", "_")
        output_path = REPORT_DIR / f"audit_report_{safe_filename}.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)

        st.caption(f"Full report saved to: {output_path}")

        json_str = json.dumps(report_data, indent=2, ensure_ascii=False)
        st.download_button(
            label="Download JSON Report",
            data=json_str,
            file_name=f"audit_{safe_filename}.json",
            mime="application/json",
        )
    else:
        st.success("Audit complete: no risks detected")
        st.caption("Document appears to comply with defined policies.")
