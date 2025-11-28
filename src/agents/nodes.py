import json
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from src.agents.state import AgentGraphState
from src.database.vector_store import retrieve_relevant_rules
from src.config import GROQ_API_KEY, LLM_MODEL

llm = ChatGroq(
    temperature=0.1,
    model_name=LLM_MODEL,
    api_key=GROQ_API_KEY,
    model_kwargs={"response_format": {"type": "json_object"}},
)


def drafter_node(state: AgentGraphState) -> AgentGraphState:
    print("Drafter agent")

    section_text = state["section_text"]
    iteration = state.get("iteration_count", 0)
    feedback = state.get("critique_feedback")

    relevant_rules = retrieve_relevant_rules(section_text)

    system_msg = """
You are a legal auditor.
Compare the clause against company policies.

Company policies:
{rules}

Output valid JSON with:
- risk_found (bool)
- risk_assessment (string)
- cited_policy (string)
"""

    human_msg = f"Clause:\n{section_text}"

    if feedback:
        print(f"Feedback received: {feedback[:50]}...")
        human_msg += (
            "\n\nPrevious analysis was rejected.\n"
            f"Critic feedback: {feedback}\n"
            "Revise your assessment accordingly."
        )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_msg),
            ("human", human_msg),
        ]
    )

    chain = prompt | llm
    response = chain.invoke({"rules": "\n- ".join(relevant_rules)})

    try:
        data = json.loads(response.content)
        risk_text = data.get("risk_assessment", "Analysis failed.")
        risk_found = data.get("risk_found", False)
    except json.JSONDecodeError:
        print("JSON parsing failed")
        risk_text = "Error parsing model output."
        risk_found = False

    print(f"Draft assessment: {risk_text[:50]}...")

    return {
        "risk_assessment": risk_text,
        "relevant_rules": relevant_rules,
        "iteration_count": iteration + 1,
        "risk_found": risk_found,
    }


def critic_node(state: AgentGraphState) -> AgentGraphState:
    print("Critic agent")

    section_text = state["section_text"]
    draft = state["risk_assessment"]
    rules = state["relevant_rules"]

    system_msg = """
You are a senior legal reviewer.

Policies:
{rules}
Clause:
{clause}
Assessment:
{draft}

Return JSON with:
- status ("APPROVED" or "REJECTED")
- feedback (string)
"""

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_msg),
            ("human", "Review this analysis."),
        ]
    )

    chain = prompt | llm
    response = chain.invoke(
        {
            "rules": "\n- ".join(rules),
            "clause": section_text,
            "draft": draft,
        }
    )

    try:
        data = json.loads(response.content)
        status = data.get("status", "REJECTED").upper()
        feedback = data.get("feedback", "No feedback provided.")
    except json.JSONDecodeError:
        print("JSON parsing failed")
        status = "REJECTED"
        feedback = "JSON error."

    is_satisfactory = status == "APPROVED"

    print(f"Decision: {status}")

    return {
        "critique_feedback": feedback,
        "is_satisfactory": is_satisfactory,
    }
