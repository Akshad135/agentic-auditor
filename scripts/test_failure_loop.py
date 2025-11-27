import sys
from pathlib import Path
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END

sys.path.append(str(Path(__file__).resolve().parent.parent))
load_dotenv()

from src.agents.state import AgentGraphState
from src.agents.nodes import critic_node
from src.agents.workflow import decide_next_step


def dumb_drafter_node(state: AgentGraphState) -> AgentGraphState:
    """Fail on first iteration to trigger critic feedback."""
    print("--- Dumb drafter agent ---")
    iteration = state.get("iteration_count", 0)
    section_text = state["section_text"]

    if iteration == 0:
        print("Acting naive: 'This clause looks fine to me.'")
        return {
            "risk_assessment": (
                "This clause is perfectly fine. "
                "50 years is a standard non-compete duration."
            ),
            "relevant_rules": ["Non-competes must not exceed 2 years."],
            "iteration_count": iteration + 1,
        }

    print("Acting careful: 'I see the issue now.'")
    return {
        "risk_assessment": (
            "HIGH RISK: The clause specifies 50 years, "
            "which strictly violates the 2-year limit."
        ),
        "relevant_rules": ["Non-competes must not exceed 2 years."],
        "iteration_count": iteration + 1,
    }


def build_test_graph():
    """Build a small StateGraph using the dumb drafter and critic."""
    workflow = StateGraph(AgentGraphState)
    workflow.add_node("drafter", dumb_drafter_node)
    workflow.add_node("critic", critic_node)
    workflow.set_entry_point("drafter")
    workflow.add_edge("drafter", "critic")
    workflow.add_conditional_edges(
        "critic",
        decide_next_step,
        {"drafter": "drafter", END: END},
    )
    return workflow.compile()


if __name__ == "__main__":
    print("Testing feedback loop (forced failure scenario)")
    app = build_test_graph()

    input_state = {
        "section_text": "The Employee agrees not to compete for 50 years.",
        "iteration_count": 0,
    }

    final = app.invoke(input_state)

    print("\nTest complete")
    print(f"Total iterations: {final['iteration_count']}")
    print(f"Final outcome: {final['is_satisfactory']}")
