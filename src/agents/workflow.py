from langgraph.graph import StateGraph, END
from src.agents.state import AgentGraphState
from src.agents.nodes import drafter_node, critic_node
from src.config import MAX_RETRIES


def decide_next_step(state: AgentGraphState):
    """
    Decide whether to end or loop back to the drafter.
    """
    is_satisfactory = state.get("is_satisfactory", False)
    iteration = state.get("iteration_count", 0)

    if is_satisfactory:
        print("Decision: approved. Ending.")
        return END

    if iteration >= MAX_RETRIES:
        print(f"Decision: max retries ({MAX_RETRIES}) reached. Ending.")
        return END

    print("Decision: rejected. Looping.")
    return "drafter"


def build_agent_graph():
    workflow = StateGraph(AgentGraphState)

    workflow.add_node("drafter", drafter_node)
    workflow.add_node("critic", critic_node)

    workflow.set_entry_point("drafter")
    workflow.add_edge("drafter", "critic")

    workflow.add_conditional_edges(
        "critic",
        decide_next_step,
        {
            "drafter": "drafter",
            END: END,
        },
    )

    return workflow.compile()
