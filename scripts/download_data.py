import pandas as pd
from datasets import load_dataset
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RAW_DATA_PATH = DATA_DIR / "real_legal_clauses.csv"

LABEL_MAP = {
    0: "Limitation of Liability",
    1: "Unilateral Change",
    2: "Content Removal",
    3: "Contract Termination",
    4: "Choice of Law / Jurisdiction",
    5: "Arbitration",
    6: "Unilateral Termination",
    7: "Exclusion of Liability",
}


def download_legal_data():
    print("Downloading LexGLUE dataset (unfair_tos)")

    try:
        dataset = load_dataset(
            "lex_glue",
            "unfair_tos",
            split="train",
            trust_remote_code=True,
        )

        print(f"Entries loaded: {len(dataset)}")
        print("Filtering unfair clauses")

        extracted_clauses = []

        for entry in dataset:
            text = entry["text"]
            labels = entry["labels"]

            if not labels:
                continue

            for label_id in labels:
                category = LABEL_MAP.get(label_id, "General Risk")
                clean_text = text.replace("\n", " ").strip()

                if len(clean_text) > 50:
                    extracted_clauses.append(
                        {
                            "category": category,
                            "text": clean_text,
                            "source": "LexGLUE_UnfairToS",
                        }
                    )

        df = pd.DataFrame(extracted_clauses)
        df = df.drop_duplicates(subset=["text"])

        df.to_csv(RAW_DATA_PATH, index=False)

        print(f"Clauses saved: {len(df)}")
        print(f"Output: {RAW_DATA_PATH}")

    except Exception as e:
        print(f"Download failed: {e}")


if __name__ == "__main__":
    download_legal_data()
