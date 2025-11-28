# Agentic Auditor 🕵️‍♂️

Agentic Auditor is an AI-driven contract reviewer that runs a Drafter–Critic agent loop to identify risks and unfair clauses in legal documents quickly and with reduced hallucination.

---

## Key features

- Drafter & Critic agent loop to reduce hallucination
- PDF ingestion and chunking
- Local policy retrieval via Qdrant vector DB
- JSON report + visual risk log output
- Self-healing: auto-downloads data and rebuilds DB if missing

---

## Architecture / Tech stack

- Frontend: Streamlit
- Orchestration: LangGraph + Groq (GPT OSS)
- Vector DB / Memory: Qdrant
- Ingestion: Unstructured + Poppler
- Python 3.10+

---

## Quickstart

1. Clone the repo:

```bash
git clone https://github.com/Akshad135/agentic-auditor
cd agentic-auditor
```

2. Create and activate a virtual environment

macOS / Linux:

```bash
python -m venv .venv
source .venv/bin/activate
```

Windows (PowerShell):

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. (Optional, GPU users) Restore CUDA PyTorch after dependency install (Poppler tends to reinstall torch without CUDA):

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 --force-reinstall
```

---

## Prerequisites

- Python 3.10+
- Poppler (required for PDF processing)

Install Poppler:

- macOS:

```bash
brew install poppler
```

- Ubuntu / Debian:

```bash
sudo apt-get install poppler-utils
```

- Windows:
  Download a Poppler binary for Windows [like this](https://github.com/oschwartz10612/poppler-windows) and add its `bin/` folder to your PATH. Restart your terminal/IDE after updating PATH.

---

## Configuration

Copy the example environment file and add your Groq API key:

```bash
cp .env.example .env
# Edit .env and set GROQ_API_KEY and other variables as needed
```

---

## Running the app

Start the Streamlit dashboard:

```bash
streamlit run app.py
```

Upload a PDF or paste text in the UI. The Drafter and Critic will debate assessments in the sidebar and the app will generate a structured JSON report and a visual risk log.

---

## Useful scripts

- System checks (GPU, Poppler, API)

```bash
python scripts/check.py
```

- Rebuild vector DB and playbook

```bash
python development/setup_db.py
```

- Run CLI audit

```bash
python development/run_audit.py
```

---

## Project layout

```bash
app.py                     # Main Streamlit dashboard
src/agents/                # LangGraph agent logic (Drafter / Critic)
data/                      # Vector DB, raw PDFs, and artifacts
development/               # Scripts: build DB, CLI, data download
requirements.txt
.env.example
```

---

## Notes

- Designed to auto-bootstrap: on first run it will attempt to download training data, derive policies, and build the vector index if missing.
- Reports are produced as structured JSON plus an interactive risk log for review.

---
