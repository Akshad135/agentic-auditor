from typing import TypedDict, List, Optional

class AgentGraphState(TypedDict):
    """
    Defines the structure of data passed between the Drafter and Critic agents.
    This acts as the 'Short-Term Memory' of the debate.
    """
    # --- INPUT ---
    section_text: str

    # --- DRAFTER OUTPUTS ---
    risk_assessment: Optional[str]
    relevant_rules: List[str]

    # --- CRITIC OUTPUTS ---
    critique_feedback: Optional[str]
    is_satisfactory: bool

    # --- METADATA ---
    iteration_count: int