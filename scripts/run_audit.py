import sys
import json
import os
from pathlib import Path
from dotenv import load_dotenv

sys.path.append(str(Path(__file__).resolve().parent.parent))
load_dotenv()

from src.ingestion.pdf_parser import parse_pdf
from src.agents.workflow import build_agent_graph

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RAW_PDF_DIR = DATA_DIR / "raw_pdfs"
REPORT_DIR = DATA_DIR / "audit_reports"

os.makedirs(REPORT_DIR, exist_ok=True)

INPUT_FILENAME = "test_contract.pdf"
INPUT_PDF_PATH = RAW_PDF_DIR / INPUT_FILENAME


def main():
    print(f"Starting audit: {INPUT_FILENAME}")

    if not INPUT_PDF_PATH.exists():
        print(f"File not found: {INPUT_PDF_PATH}")
        return

    chunks = parse_pdf(str(INPUT_PDF_PATH))
    print(f"Chunks loaded: {len(chunks)}")

    app = build_agent_graph()
    final_report = []

    print("Running analysis")

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
        risk_found_bool = result.get("risk_found", False)

        is_risk = False

        if risk_found_bool:
            is_risk = True
        else:
            # Fallback text-based check
            upper_text = assessment.upper()
            danger_signs = ["HIGH RISK", "POLICY VIOLATION", "NON-COMPLIANT"]
            negations = ["NO VIOLATION", "COMPLIANT", "DOES NOT VIOLATE", "SAFE"]

            has_danger = any(sign in upper_text for sign in danger_signs)
            is_negated = any(neg in upper_text for neg in negations)

            if has_danger and not is_negated:
                print(f"Override triggered: {assessment[:40]}...")
                is_risk = True

        if is_risk:
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

    with open(output_path, "w") as f:
        json.dump(final_report, f, indent=2)

    print(f"Audit complete. Risks found: {len(final_report)}")
    print(f"Report saved to: {output_path}")


if __name__ == "__main__":
    main()
