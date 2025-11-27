import sys
from pathlib import Path
from dotenv import load_dotenv

sys.path.append(str(Path(__file__).resolve().parent.parent))
load_dotenv()

from src.agents.state import AgentGraphState
from src.agents.nodes import drafter_node, critic_node


def test_pipeline():
    print("Running component test")

    bad_clause = (
        "The Employee agrees not to compete with the Company "
        "for a period of 5 years after termination."
    )

    initial_state: AgentGraphState = {
        "section_text": bad_clause,
        "iteration_count": 0,
        "risk_assessment": None,
        "relevant_rules": [],
        "critique_feedback": None,
        "is_satisfactory": False,
    }

    # Test Drafter
    print("\nTesting drafter node")
    try:
        draft_result = drafter_node(initial_state)
        print("Drafter completed")
        print(f"Assessment preview: {draft_result['risk_assessment'][:100]}...")

        if draft_result["relevant_rules"]:
            print(f"Rules retrieved: {len(draft_result['relevant_rules'])}")
        else:
            print("Warning: No rules retrieved")

    except Exception as e:
        print(f"Drafter failed: {e}")
        return

    # Test Critic
    print("\nTesting critic node")
    try:
        state_for_critic = {**initial_state, **draft_result}
        critic_result = critic_node(state_for_critic)

        decision = "Approved" if critic_result["is_satisfactory"] else "Rejected"
        print("Critic completed")
        print(f"Decision: {decision}")
        print(f"Feedback: {critic_result['critique_feedback']}")

    except Exception as e:
        print(f"Critic failed: {e}")


if __name__ == "__main__":
    test_pipeline()
