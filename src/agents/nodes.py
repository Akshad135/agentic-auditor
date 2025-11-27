import os
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from src.agents.state import AgentGraphState
from src.database.vector_store import retrieve_relevant_rules
from src.config import GROQ_API_KEY, LLM_MODEL

# Initialize the LLM once
llm = ChatGroq(
    temperature=0.1, # Low temperature for factual legal analysis
    model_name=LLM_MODEL,
    api_key=GROQ_API_KEY
)

def drafter_node(state: AgentGraphState) -> AgentGraphState:
    """
    Agent A: The Risk Hunter.
    1. Looks up policies in Qdrant.
    2. Drafts a risk assessment.
    """
    print("--- 🕵️‍♂️ DRAFTER AGENT WORKING ---")
    section_text = state["section_text"]
    iteration = state.get("iteration_count", 0)

    # 1. RAG: Get the Ground Truth from our Local DB
    # We use the raw text as the query
    relevant_rules = retrieve_relevant_rules(section_text)
    
    # 2. The Prompt
    system_msg = """You are a cynical Legal Auditor AI. 
    Your goal is to detect risks in contract clauses based STRICTLY on the provided Company Policies.
    
    Company Policies:
    {rules}
    
    Instructions:
    - Compare the 'Clause Under Review' against the 'Company Policies'.
    - If the clause violates a policy, flag it as HIGH RISK.
    - If the clause is safe, mark it as LOW RISK.
    - Cite the specific policy rule you are relying on.
    """
    
    human_msg = f"Clause Under Review:\n{section_text}"

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_msg),
        ("human", human_msg)
    ])

    # 3. Invoke LLM
    chain = prompt | llm
    response = chain.invoke({"rules": "\n- ".join(relevant_rules)})
    
    print(f"   📝 Draft Generated (Length: {len(response.content)})")

    # 4. Update State
    return {
        "risk_assessment": response.content,
        "relevant_rules": relevant_rules,
        "iteration_count": iteration + 1
    }

def critic_node(state: AgentGraphState) -> AgentGraphState:
    """
    Agent B: The Senior Partner.
    1. Reviews the Drafter's work.
    2. Checks for hallucinations or missed loopholes.
    """
    print("--- ⚖️ CRITIC AGENT WORKING ---")
    section_text = state["section_text"]
    draft = state["risk_assessment"]
    rules = state["relevant_rules"]
    
    # 1. The Prompt
    system_msg = """You are a Senior Legal Partner. Your job is to grade the work of a Junior Auditor.
    
    The Junior Auditor was given this Ground Truth:
    {rules}
    
    And they analyzed this Clause:
    {clause}
    
    Their Analysis:
    {draft}
    
    Your Task:
    - Did the Junior Auditor hallucinate a rule that doesn't exist?
    - Did they miss a clear violation mentioned in the Ground Truth?
    - Reply with 'APPROVED' if the analysis is solid.
    - Reply with 'REJECTED' followed by your critique if it is flawed.
    """

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_msg),
        ("human", "Grade this analysis.")
    ])

    # 2. Invoke LLM
    chain = prompt | llm
    # We format the rules into a string list for the prompt
    response = chain.invoke({
        "rules": "\n- ".join(rules),
        "clause": section_text,
        "draft": draft
    })

    content = response.content
    is_satisfactory = "APPROVED" in content.upper()
    
    print(f"   👩‍⚖️ Critique: {'✅ APPROVED' if is_satisfactory else '❌ REJECTED'}")

    return {
        "critique_feedback": content,
        "is_satisfactory": is_satisfactory
    }