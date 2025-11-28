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
    iteration = state.get("iteration_count", 0)
    print(f"\n--- DRAFTER | Iteration {iteration} ---")

    section_text = state["section_text"]
    feedback = state.get("critique_feedback")

    relevant_rules = retrieve_relevant_rules(section_text)

    system_msg = """
                You are a legal auditor.
                Compare the input clause against the provided company policies.

                Company policies:
                {rules}

                Your response must be a valid JSON object. Do not include any explanation or text outside the JSON.

                Required JSON Structure:
                {{
                    "risk_found": boolean,
                    "risk_assessment": "string (brief explanation)",
                    "cited_policy": "string (policy name or None)"
                }}
                """

    human_msg = f"Clause:\n{section_text}"

    if feedback:
        print(f"[DRAFTER] Feedback received: {feedback[:60]}...")
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
        print("[DRAFTER] JSON parsing failed")
        risk_text = "Error parsing model output."
        risk_found = False

    print(f"[DRAFTER] Risk found: {risk_found}")
    print(f"[DRAFTER] Assessment: {risk_text[:120]}...")

    return {
        "risk_assessment": risk_text,
        "relevant_rules": relevant_rules,
        "iteration_count": iteration + 1,
        "risk_found": risk_found,
    }


def critic_node(state: AgentGraphState) -> AgentGraphState:
    print("\n--- CRITIC ---")

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

                Your response must be a valid JSON object. Do not include any explanation or text outside the JSON.

                Required JSON Structure:
                {{
                    "status": "APPROVED" or "REJECTED",
                    "feedback": "string (explanation of decision)"
                }}
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
        print("[CRITIC] JSON parsing failed")
        status = "REJECTED"
        feedback = "JSON error."

    is_satisfactory = status == "APPROVED"

    print(f"[CRITIC] Decision: {status}")
    print(f"[CRITIC] Feedback: {feedback[:120]}...")

    return {
        "critique_feedback": feedback,
        "is_satisfactory": is_satisfactory,
    }