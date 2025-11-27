import sys
import os
from pathlib import Path
from dotenv import load_dotenv

# Setup path
sys.path.append(str(Path(__file__).resolve().parent.parent))

# Load Env (API Keys)
load_dotenv()

from src.agents.state import AgentGraphState
from src.agents.nodes import drafter_node, critic_node

def test_pipeline():
    print("🧪 STARTING COMPONENT TEST")
    
    # 1. Define a Mock Input (A bad clause that violates our rules)
    # Our rule says "Max 2 years". This clause says "5 years".
    bad_clause = "The Employee agrees not to compete with the Company for a period of 5 years after termination."
    
    initial_state: AgentGraphState = {
        "section_text": bad_clause,
        "iteration_count": 0,
        "risk_assessment": None,
        "relevant_rules": [],
        "critique_feedback": None,
        "is_satisfactory": False
    }

    # 2. Test Drafter (Agent A)
    print("\n\n>>> TESTING DRAFTER NODE...")
    try:
        draft_result = drafter_node(initial_state)
        print("✅ Drafter Success!")
        print(f"   Assessment Snippet: {draft_result['risk_assessment'][:100]}...")
        
        # Check if RAG worked
        if draft_result['relevant_rules']:
            print(f"   RAG Context Found: {len(draft_result['relevant_rules'])} rules")
        else:
            print("   ⚠️ WARNING: No rules retrieved from Qdrant.")
            
    except Exception as e:
        print(f"❌ Drafter Failed: {e}")
        return

    # 3. Test Critic (Agent B)
    # We feed the OUTPUT of the Drafter into the INPUT of the Critic
    print("\n\n>>> TESTING CRITIC NODE...")
    try:
        # Merge the results
        state_for_critic = {**initial_state, **draft_result}
        
        critic_result = critic_node(state_for_critic)
        print("✅ Critic Success!")
        print(f"   Decision: {'Approved' if critic_result['is_satisfactory'] else 'Rejected'}")
        print(f"   Feedback: {critic_result['critique_feedback']}")
        
    except Exception as e:
        print(f"❌ Critic Failed: {e}")

if __name__ == "__main__":
    test_pipeline()