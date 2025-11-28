import sys
import json
import os
from pathlib import Path
from dotenv import load_dotenv
from qdrant_client import QdrantClient

sys.path.append(str(Path(__file__).resolve().parent.parent))
load_dotenv()

from src.ingestion.pdf_parser import parse_pdf
from src.agents.workflow import build_agent_graph

from src.config import (
    DATA_DIR,
    RAW_PDF_DIR,
    REPORT_DIR,
    VECTOR_DB_PATH,
    COLLECTION_NAME,
    DERIVED_PLAYBOOK_JSON,
    RAW_DATA_CSV,
    INPUT_PDF_PATH,
    INPUT_FILENAME,
)

from development.derive_playbook import derive_policies
from development.download_data import download_legal_data
from development.setup_db import initialize_memory

os.makedirs(REPORT_DIR, exist_ok=True)
os.makedirs(RAW_PDF_DIR, exist_ok=True)


def check_system_health():
    """Ensure vector DB and playbook are ready. Auto-repair if needed."""
    print("Running system health check")

    db_ready = False

    if VECTOR_DB_PATH.exists():
        client = QdrantClient(path=str(VECTOR_DB_PATH))
        collections = [c.name for c in client.get_collections().collections]

        if COLLECTION_NAME in collections:
            try:
                count = client.count(COLLECTION_NAME).count
                if count > 0:
                    print(f"Vector DB ready ({count} policies)")
                    db_ready = True
            except Exception:
                print("Vector DB corrupted or empty")

        client.close()

    if db_ready:
        return

    print("Vector DB missing or empty. Running setup")

    if not DERIVED_PLAYBOOK_JSON.exists():
        print("Playbook missing")

        if not RAW_DATA_CSV.exists():
            print("Raw dataset missing. Downloading")
            download_legal_data()

        print("Deriving new policies")
        derive_policies()

    print("Indexing policies into vector DB")
    initialize_memory()
    print("System ready")


def main():
    check_system_health()

    print(f"Starting audit: {INPUT_FILENAME}")

    if not INPUT_PDF_PATH.exists():
        print(f"File not found: {INPUT_PDF_PATH}")
        print("Place it in data/raw_pdfs/")
        return

    chunks = parse_pdf(str(INPUT_PDF_PATH))
    print(f"Chunks loaded: {len(chunks)}")

    app = build_agent_graph()
    final_report = []

    print("Running agent analysis")

    for i, chunk in enumerate(chunks):
        if len(chunk) < 30:
            continue

        print(f"Processing chunk {i + 1}")

        input_state = {
            "section_text": chunk,
            "iteration_count": 0,
        }

        result = app.invoke(input_state)

        assessment = result.get("risk_assessment", "")
        risk_found = result.get("risk_found", False)

        if not risk_found:
            upper_text = assessment.upper()
            danger = ["HIGH RISK", "POLICY VIOLATION"]
            safe = ["NO VIOLATION", "COMPLIANT"]

            if any(d in upper_text for d in danger) and not any(s in upper_text for s in safe):
                risk_found = True

        if risk_found:
            print(f"Risk detected: {assessment[:80]}...")
            final_report.append(
                {
                    "chunk_id": i,
                    "original_text": chunk,
                    "risk_analysis": assessment,
                    "iterations": result["iteration_count"],
                }
            )
        else:
            print(f"No risk: {assessment[:50]}...")

    output_filename = f"{INPUT_PDF_PATH.stem}_audit.json"
    output_path = REPORT_DIR / output_filename

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=2, ensure_ascii=False)

    print(f"Audit complete. Risks found: {len(final_report)}")
    print(f"Report saved: {output_path}")


if __name__ == "__main__":
    main()
