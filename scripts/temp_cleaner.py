import sys
import json
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.utils.text_cleaner import clean_text
from src.config import DATA_DIR

PLAYBOOK_PATH = DATA_DIR / "derived_playbook.json"


def clean_playbook_file():
    print("Cleaning playbook")

    if not PLAYBOOK_PATH.exists():
        print(f"Playbook not found: {PLAYBOOK_PATH}")
        return

    try:
        with open(PLAYBOOK_PATH, "r") as f:
            policies = json.load(f)
    except json.JSONDecodeError:
        print("Failed to load JSON")
        return

    print(f"Entries found: {len(policies)}")

    cleaned_policies = []
    for policy in policies:
        policy["policy_name"] = clean_text(policy.get("policy_name", ""))
        policy["text"] = clean_text(policy.get("text", ""))
        cleaned_policies.append(policy)

    with open(PLAYBOOK_PATH, "w") as f:
        json.dump(cleaned_policies, f, indent=2, ensure_ascii=False)

    print("Playbook cleaned and overwritten")


if __name__ == "__main__":
    clean_playbook_file()
