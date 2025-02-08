from dotenv import load_dotenv
import os
from typing import TypedDict, List, Literal, Annotated
from langgraph.graph import StateGraph, END
from langchain_community.document_loaders import Docx2txtLoader
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
import operator
import datetime

# Load the .env file
load_dotenv()

# Access the API key
api_key = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_API_KEY"] = api_key




# Define state structure
class AgentState(TypedDict):
    conversation_history: Annotated[List[str], operator.add]
    test_report: str
    generated_questions: List[str]
    pending_questions: List[str]
    last_follow_up: datetime.datetime
    symptoms_collected: bool
    report_processed: bool
    next_action: Literal[
        "collect_symptoms", 
        "process_report",
        "clarify_questions",
        "follow_up",
        "exit"
    ]
    user_input: str

# Initialize LLM
llm = ChatOpenAI(temperature=0.2, model="gpt-4-turbo")

# Define nodes with error handling
def supervisor_node(state: AgentState):
    try:
        # Ensure conversation history exists
        if not state.get("conversation_history"):
            state["conversation_history"] = ["Assistant: Hello! I'm your health assistant. Let's start with your symptoms."]
        
        messages = [
            SystemMessage(content="""You are a medical workflow supervisor. Decide next action:
            1. collect_symptoms: If symptoms not fully collected
            2. process_report: If test report uploaded but not processed
            3. clarify_questions: If pending questions from report
            4. follow_up: If time for regular check-in
            5. exit: Only when consultation complete
            
            Current State:
            Symptoms Collected: {collected}
            Test Report: {report}
            Pending Questions: {questions}
            Conversation Length: {length}""".format(
                collected=state.get("symptoms_collected", False),
                report="Uploaded" if state.get("test_report") else "None",
                questions=len(state.get("pending_questions", [])),
                length=len(state.get("conversation_history", []))
            )),
            HumanMessage(content="Recent conversation:\n" + "\n".join(state["conversation_history"][-3:]))
        ]
        
        decision = llm.invoke(messages).content.lower().strip()
        return {"next_action": decision}
    
    except Exception as e:
        print(f"Supervisor error: {str(e)}")
        return {"next_action": "collect_symptoms"}

def handle_symptoms(state: AgentState):
    try:
        messages = [
            SystemMessage(content="""You are a medical assistant. Your tasks:
            1. Ask specific symptom questions
            2. Request details about duration, intensity, location
            3. Ask one question at a time
            4. Maintain professional but friendly tone"""),
            HumanMessage(content=f"Conversation History:\n{state.get('conversation_history', [])}\nPatient Input: {state.get('user_input', '')}")
        ]
        
        response = llm.invoke(messages).content
        return {
            "conversation_history": [
                f"Patient: {state.get('user_input', '')}",
                f"Assistant: {response}"
            ],
            "next_action": "supervisor",
            "symptoms_collected": len(state.get("conversation_history", [])) > 5
        }
    except Exception as e:
        print(f"Symptom collection error: {str(e)}")
        return {"next_action": "supervisor"}

# Add other nodes (process_test_report, clarify_questions, follow_up) from previous version
# with similar error handling

def generate_summary(state: AgentState):
    try:
        messages = [
            SystemMessage(content="""Create a clinical summary:
            1. Organize symptoms chronologically
            2. Highlight key findings
            3. Note patient responses
            4. Format: Symptoms, Findings, Notes"""),
            HumanMessage(content="\n".join(state.get("conversation_history", [])))
        ]
        return llm.invoke(messages).content
    except Exception as e:
        return f"Error generating summary: {str(e)}"

# Build workflow
workflow = StateGraph(AgentState)

# Add nodes
nodes = {
    "supervisor": supervisor_node,
    "collect_symptoms": handle_symptoms,
    # Add other nodes here
}

for name, node in nodes.items():
    workflow.add_node(name, node)

# Define edges
workflow.add_conditional_edges(
    "supervisor",
    lambda state: state.get("next_action", "collect_symptoms"),
    {
        "collect_symptoms": "collect_symptoms",
        "process_report": "process_report", 
        "clarify_questions": "clarify_questions",
        "follow_up": "follow_up",
        "exit": END
    }
)

for node in ["collect_symptoms", "process_report", "clarify_questions", "follow_up"]:
    workflow.add_edge(node, "supervisor")

workflow.set_entry_point("supervisor")
agent = workflow.compile()

# Chat interface with improved error handling
def chat_interface():
    state = {
        "conversation_history": ["Assistant: Hello! I'm your health assistant. Let's start with your symptoms."],
        "test_report": "",
        "generated_questions": [],
        "pending_questions": [],
        "last_follow_up": None,
        "symptoms_collected": False,
        "report_processed": False,
        "next_action": "collect_symptoms",
        "user_input": ""
    }
    
    print(state["conversation_history"][0])
    
    while True:
        try:
            if state["next_action"] == "process_report":
                report_path = input("\n[System] Please upload test report path: ")
                state["test_report"] = report_path
                state["user_input"] = "[REPORT_UPLOADED]"
            else:
                user_input = input("\nPatient: ")
                state["user_input"] = user_input
            
            result = agent.invoke(state)
            state.update(result)
            
            if state["conversation_history"]:
                print(f"\nAssistant: {state['conversation_history'][-1].split('Assistant: ')[-1]}")
            
            if state.get("next_action") == "exit":
                summary = generate_summary(state)
                print(f"\nConsultation Summary for Doctor:\n{summary}")
                break
                
        except Exception as e:
            print(f"Error in conversation: {str(e)}")
            state["next_action"] = "supervisor"

if __name__ == "__main__":
    chat_interface()