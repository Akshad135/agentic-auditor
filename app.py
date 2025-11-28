import streamlit as st
import os
import json
import shutil
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

from src.ingestion.pdf_parser import parse_pdf
from src.agents.workflow import build_agent_graph
from src.config import DATA_DIR, REPORT_DIR

# Setup Directories
TEMP_DIR = DATA_DIR / "temp_uploads"
os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

st.set_page_config(page_title="Legal Audit Dashboard", layout="wide")
st.markdown("""
    <style>
    .stTextArea textarea { font-family: monospace; }
    
    .risk-box { 
        padding: 1rem; 
        border-left: 5px solid #ff4b4b; 
        background-color: #262730;  /* Dark background to match Streamlit Dark Mode */
        color: #ffffff;             /* White text */
        margin-bottom: 1rem;
        border-radius: 4px;
    }
    
    .safe-box { 
        padding: 1rem; 
        border-left: 5px solid #4caf50; 
        background-color: #262730;  /* Dark background to match Streamlit Dark Mode */
        color: #ffffff;             /* White text */
        margin-bottom: 1rem;
        border-radius: 4px;
    }
    </style>
""", unsafe_allow_html=True)

st.title("Legal Audit Dashboard")
st.markdown("---")

with st.sidebar:
    st.header("System Status")
    if (DATA_DIR / "qdrant_db").exists():
        st.success("Vector Database: Active")
    else:
        st.error("Vector Database: Missing")
    
    st.markdown("---")
    st.caption("Powered by Agentic Auditor")


input_mode = st.radio("Input Method", ["Upload Document (PDF)", "Paste Text"], horizontal=True)

chunks = []
source_name = ""

if input_mode == "Upload Document (PDF)":
    uploaded_file = st.file_uploader("Select PDF Contract", type="pdf")
    if uploaded_file:
        source_name = uploaded_file.name
        # Save temp file
        temp_path = TEMP_DIR / "current_audit.pdf"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Parse immediately
        with st.status("Ingesting document...", expanded=False) as status:
            st.write("Parsing PDF structure...")
            chunks = parse_pdf(str(temp_path))
            status.update(label="Document ready", state="complete", expanded=False)
        
        st.info(f"Loaded {len(chunks)} text segments from {source_name}")

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
    
    for i, chunk in enumerate(chunks):
        progress = (i + 1) / len(chunks)
        progress_bar.progress(progress)
        status_text.text(f"Processing segment {i+1} of {len(chunks)}...")
        
        input_state = {
            "section_text": chunk,
            "iteration_count": 0,
        }
        
        try:
            result = app.invoke(input_state)
            
            risk_found = result.get("risk_found", False)
            assessment = result.get("risk_assessment", "No assessment provided.")
            
            if risk_found:
                report_data.append({
                    "chunk_id": i + 1,
                    "text_preview": chunk[:200] + "..." if len(chunk) > 200 else chunk,
                    "assessment": assessment,
                    "iterations": result.get("iteration_count", 0)
                })
                
                with log_container:
                    st.markdown(f"""
                    <div class="risk-box">
                        <strong>RISK DETECTED (Segment {i+1})</strong><br>
                        {assessment}
                    </div>
                    """, unsafe_allow_html=True)
            else:
                with log_container:
                    st.markdown(f"""
                    <div class="safe-box">
                        <strong>Segment {i+1}: Safe</strong>
                    </div>
                    """, unsafe_allow_html=True)
                pass

        except Exception as e:
            st.error(f"Error processing segment {i+1}: {e}")

    st.markdown("---")
    st.header("Audit Report")
    
    if report_data:
        st.error(f"Audit Complete: {len(report_data)} risks identified.")
        output_path = REPORT_DIR / f"audit_report_{source_name}.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)
        
        st.caption(f"Full detailed report saved to: {output_path}")
        json_str = json.dumps(report_data, indent=2, ensure_ascii=False)
        st.download_button(
            label="Download JSON Report",
            data=json_str,
            file_name=f"audit_{source_name}.json",
            mime="application/json"
        )
    else:
        st.success("Audit Complete: No risks detected.")
        st.caption("The document appears to comply with defined policies.")