import os
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from pathlib import Path

# Define where we want to save the test PDF
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw_pdfs"
os.makedirs(DATA_DIR, exist_ok=True)
FILE_PATH = DATA_DIR / "test_contract.pdf"

def create_contract():
    c = canvas.Canvas(str(FILE_PATH), pagesize=letter)
    width, height = letter

    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, height - 50, "NON-DISCLOSURE AGREEMENT")
    
    c.setFont("Helvetica", 12)
    text_lines = [
        "This Agreement is made between 'TechCorp' and 'The Employee'.",
        "",
        "1. CONFIDENTIALITY",
        "The Employee agrees to keep all proprietary information confidential.",
        "",
        "2. NON-COMPETE (The Trap Clause)",
        "The Employee agrees not to compete with TechCorp for a period of",
        "50 years after termination of employment.",
        "",
        "3. JURISDICTION",
        "All disputes shall be resolved in the State of Mars.",
        "",
        "Signed,",
        "The CEO"
    ]

    # Draw lines downwards
    y = height - 100
    for line in text_lines:
        c.drawString(50, y, line)
        y -= 20

    c.save()
    print(f"✅ Created dummy contract at: {FILE_PATH}")

if __name__ == "__main__":
    create_contract()