import sys
from pathlib import Path
from dotenv import load_dotenv

sys.path.append(str(Path(__file__).resolve().parent.parent))
load_dotenv()

from src.agents.workflow import build_agent_graph


def run_test():
    print("Starting agent loop test")
    app = build_agent_graph()

    input_state = {
        "section_text": (
            "The Employee agrees not to compete with the Company for a "
            "period of 5 (five) years."
        ),
        "iteration_count": 0,
    }

    final_state = app.invoke(input_state)

    print("\nTest complete")
    print(f"Total iterations: {final_state['iteration_count']}")
    print(f"Final outcome: {final_state.get('is_satisfactory')}")


if __name__ == "__main__":
    run_test()
