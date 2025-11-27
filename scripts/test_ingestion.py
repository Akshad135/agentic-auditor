import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.ingestion.pdf_parser import parse_pdf
from scripts.create_dummy_pdf import create_contract, FILE_PATH


def run_eye_exam():
    print("Starting ingestion test")

    create_contract()

    print(f"Reading file: {FILE_PATH}")
    chunks = parse_pdf(str(FILE_PATH))

    print("Extracted chunks:")
    for i, chunk in enumerate(chunks):
        preview = chunk[:60] + "..." if len(chunk) > 60 else chunk
        print(f"[{i}] {preview}")

    has_trap = any("50 years" in c for c in chunks)

    if has_trap:
        print("Result: clause detected")
    else:
        print("Result: clause not detected")


if __name__ == "__main__":
    run_eye_exam()
