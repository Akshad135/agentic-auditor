from typing import TypedDict, List, Optional

class AgentGraphState(TypedDict):
    """
    Shared state passed between drafter and critic nodes.
    """
    # Input
    section_text: str

    # Drafter outputs
    risk_assessment: Optional[str]
    relevant_rules: List[str]
    risk_found: bool

    # Critic outputs
    critique_feedback: Optional[str]
    is_satisfactory: bool

    # Metadata
    iteration_count: int