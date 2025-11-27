import sys
import json
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

sys.path.append(str(Path(__file__).resolve().parent.parent))
load_dotenv()

from src.config import GROQ_API_KEY, LLM_MODEL
from src.utils.text_cleaner import clean_text

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
INPUT_CSV = DATA_DIR / "real_legal_clauses.csv"
OUTPUT_PLAYBOOK = DATA_DIR / "derived_playbook.json"


def derive_policies():
    print("Starting policy derivation")

    if not INPUT_CSV.exists():
        print("Input CSV not found")
        return

    df = pd.read_csv(INPUT_CSV)
    print(f"Rows loaded: {len(df)}")

    df["text"] = df["text"].apply(clean_text)

    categories = df["category"].unique()
    print(f"Categories: {len(categories)} (2 rules each)")

    llm = ChatGroq(
        temperature=0.3,
        model_name=LLM_MODEL,
        api_key=GROQ_API_KEY,
        model_kwargs={"response_format": {"type": "json_object"}},
    )

    derived_playbook = []

    for category in categories:
        print(f"Processing: {category}")

        examples = (
            df[df["category"] == category]["text"]
            .sample(n=min(8, len(df)))
            .tolist()
        )

        examples_text = "\n".join(f"- {ex}" for ex in examples)

        system_msg = """
You are the General Counsel.

From the provided bad contract clauses, derive two defensive company policies.
Return only valid JSON in the format:

{
  "rules": [
    {"name": "...", "text": "..."},
    {"name": "...", "text": "..."}
  ]
}
"""

        human_msg = f"""
CATEGORY: {category}

BAD EXAMPLES:
{examples_text}
"""

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_msg),
                ("human", human_msg),
            ]
        )

        try:
            chain = prompt | llm
            response = chain.invoke({})
            data = json.loads(response.content)

            rules = data.get("rules", [])

            for rule in rules:
                rule_name = clean_text(rule.get("name", ""))
                rule_text = clean_text(rule.get("text", ""))

                print(f"  Derived: {rule_name}")

                derived_playbook.append(
                    {
                        "policy_name": rule_name,
                        "category": category,
                        "text": rule_text,
                    }
                )

        except Exception as e:
            print(f"  Failed for {category}: {e}")

    with open(OUTPUT_PLAYBOOK, "w") as f:
        json.dump(derived_playbook, f, indent=2)

    print(f"Policies generated: {len(derived_playbook)}")
    print(f"Saved to: {OUTPUT_PLAYBOOK}")


if __name__ == "__main__":
    derive_policies()
