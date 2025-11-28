import sys
import json
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

sys.path.append(str(Path(__file__).resolve().parent.parent))
load_dotenv()

from src.config import GROQ_API_KEY, LLM_MODEL, RAW_DATA_CSV, DERIVED_PLAYBOOK_JSON
from src.utils.text_cleaner import clean_text


def derive_policies():
    print("Starting policy derivation")

    if not RAW_DATA_CSV.exists():
        print(f"Input CSV not found: {RAW_DATA_CSV}")
        return

    df = pd.read_csv(RAW_DATA_CSV)
    print(f"Rows loaded: {len(df)}")

    df["text"] = df["text"].apply(clean_text)

    categories = df["category"].unique()
    print(f"Categories: {len(categories)}")

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
Return only valid JSON:

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

            for rule in data.get("rules", []):
                rule_name = clean_text(rule.get("name", ""))
                rule_text = clean_text(rule.get("text", ""))

                print(f"Derived: {rule_name}")

                derived_playbook.append(
                    {
                        "policy_name": rule_name,
                        "category": category,
                        "text": rule_text,
                    }
                )

        except Exception as e:
            print(f"Failed for {category}: {e}")

    with open(DERIVED_PLAYBOOK_JSON, "w") as f:
        json.dump(derived_playbook, f, indent=2)

    print(f"Policies generated: {len(derived_playbook)}")
    print(f"Saved to: {DERIVED_PLAYBOOK_JSON}")


if __name__ == "__main__":
    derive_policies()
