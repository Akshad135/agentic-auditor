import os
import json
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from src.agents.state import AgentGraphState
from src.database.vector_store import retrieve_relevant_rules
from src.config import GROQ_API_KEY, LLM_MODEL

# Initialize LLM with JSON Mode forced
llm = ChatGroq(
    temperature=0.1,
    model_name=LLM_MODEL,
    api_key=GROQ_API_KEY,
    model_kwargs={"response_format": {"type": "json_object"}},
)

def drafter_node(state: AgentGraphState) -> AgentGraphState:
    print("--- 🕵️‍♂️ DRAFTER AGENT (JSON MODE) ---")
    section_text = state["section_text"]
    iteration = state.get("iteration_count", 0)
    feedback = state.get("critique_feedback")

    # 1. RAG retrieval
    relevant_rules = retrieve_relevant_rules(section_text)

    # 2. Dynamic prompt construction
    system_msg = """You are a Legal Auditor AI.
                Compare the contract clause against the company policies.

                Company policies:
                {rules}

                Output strictly in valid JSON with these keys:
                - "risk_found": (boolean) True if violated.
                - "risk_assessment": (string) Brief explanation of the risk.
                - "cited_policy": (string) The specific policy text used for the finding.
                """
    human_msg = f"Clause under review:\n{section_text}"

    # Add critic feedback on retries
    if feedback:
        print(f"   🔄 DETECTED FEEDBACK: {feedback[:50]}...")
        human_msg += (
            "\n\nIMPORTANT: Your previous analysis was rejected.\n"
            f"Critic feedback: {feedback}\n\n"
            "Refine your assessment to address this feedback."
        )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_msg),
            ("human", human_msg),
        ]
    )

    # 3. Invoke and parse
    chain = prompt | llm
    response = chain.invoke({"rules": "\n- ".join(relevant_rules)})

    try:
        data = json.loads(response.content)
        risk_text = data.get("risk_assessment", "Analysis failed.")
        risk_found = data.get("risk_found", False)
    except json.JSONDecodeError:
        print("❌ JSON parsing failed. Raw output:", response.content)
        risk_text = "Error parsing model output."
        risk_found = False

    print(f"   📝 Draft assessment: {risk_text[:50]}...")

    return {
        "risk_assessment": risk_text,
        "relevant_rules": relevant_rules,
        "iteration_count": iteration + 1,
        "risk_found": risk_found
    }

def critic_node(state: AgentGraphState) -> AgentGraphState:
    print("--- ⚖️ CRITIC AGENT (JSON MODE) ---")
    section_text = state["section_text"]
    draft = state["risk_assessment"]
    rules = state["relevant_rules"]

    system_msg = """You are a senior legal partner.
                Review the junior auditor's assessment.

                Ground truth: {rules}
                Clause: {clause}
                Assessment: {draft}

                Output strictly in valid JSON with these keys:
                - "status": (string) "APPROVED" or "REJECTED"
                - "feedback": (string) Explanation of why.
                """

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_msg),
            ("human", "Grade this analysis."),
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
        print("❌ JSON parsing failed.")
        status = "REJECTED"
        feedback = "JSON error."

    is_satisfactory = status == "APPROVED"

    print(f"   👩‍⚖️ Decision: {status}")

    return {
        "critique_feedback": feedback,
        "is_satisfactory": is_satisfactory,
    }