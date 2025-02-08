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

# Initialize LLMs
llm = ChatOpenAI(temperature=0.2, model="gpt-4o-mini")

def supervisor_node(state: AgentState):
    try:
        # Initialize conversation history if empty
        if not state.get("conversation_history"):
            state["conversation_history"] = ["Assistant: Hello! I'm your health assistant. Let's start with your symptoms."]
            
        # Check if exit condition is met
        if state.get("symptoms_collected", False) and state.get("report_processed", False) and not state.get("pending_questions"):
            print("[Supervisor] Consultation is complete. Exiting.")
            return {"next_action": "exit"}

        # Get last 3 messages safely
        last_messages = state["conversation_history"][-3:] if len(state["conversation_history"]) >= 3 else state["conversation_history"]
        
        messages = [
            SystemMessage(content=f"""You are a medical workflow supervisor. Decide next action:
            1. collect_symptoms: If symptoms not fully collected
            2. process_report: If test report uploaded but not processed
            3. clarify_questions: If pending questions from report
            4. follow_up: If time for regular check-in
            5. exit: Only when consultation complete
            
            Current State:
            Symptoms Collected: {state.get("symptoms_collected", False)}
            Test Report: {'Uploaded' if state.get('test_report') else 'None'}
            Pending Questions: {len(state.get('pending_questions', []))}
            Conversation Length: {len(state.get('conversation_history', []))}"""),
            HumanMessage(content="Recent conversation:\n" + "\n".join(last_messages))
        ]
        
        decision = llm.invoke(messages).content.lower().strip()
        valid_actions = ["collect_symptoms", "process_report", "clarify_questions", "follow_up", "exit"]
        
        if decision not in valid_actions:
            decision = "collect_symptoms"  # Default action to avoid infinite loop

        print(f"[Supervisor] Decision: {decision}")  # Debugging line

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

def generate_summary(state: AgentState):
    messages = [
        SystemMessage(content="""Create a clinical summary for the doctor:
        1. Organize symptoms chronologically
        2. Highlight key findings from test reports
        3. Note patient responses to clarification questions
        4. Format with sections: Symptoms, Test Findings, Important Notes"""),
        HumanMessage(content="\n".join(state["conversation_history"]))
    ]
    return summary_llm.invoke(messages).content


def process_test_report(state: AgentState):
    try:
        loader = Docx2txtLoader(state["test_report"])
        docs = loader.load()
        report_content = docs[0].page_content
        
        messages = [
            SystemMessage(content="""Analyze this test report and generate specific yes/no questions 
            to verify patient experiences. Format each question as '- [finding]: [question]'"""),
            HumanMessage(content=report_content)
        ]
        
        questions = analysis_llm.invoke(messages).content
        return {
            "generated_questions": [q.strip() for q in questions.split("\n") if q.strip()],
            "pending_questions": [q.strip() for q in questions.split("\n") if q.strip()],
            "test_report": "",  # Reset after processing
            "next_action": "supervisor"
        }
    except Exception as e:
        return {
            "conversation_history": [f"Error processing report: {str(e)}"],
            "next_action": "supervisor"
        }

def clarify_questions(state: AgentState):
    if not state["pending_questions"]:
        return {"next_action": "supervisor"}
    
    current_question = state["pending_questions"][0]
    messages = [
        SystemMessage(content="Analyze patient's response to the medical question."),
        HumanMessage(content=f"""Question: {current_question}
        Patient Response: {state['user_input']}
        Provide 1-sentence analysis:""")
    ]
    
    analysis = analysis_llm.invoke(messages).content
    return {
        "conversation_history": [
            f"Asked: {current_question}",
            f"Patient: {state['user_input']}",
            f"Analysis: {analysis}"
        ],
        "pending_questions": state["pending_questions"][1:],
        "next_action": "supervisor"
    }

def follow_up(state: AgentState):
    messages = [
        SystemMessage(content="""Generate follow-up questions based on:
        - Conversation history
        - Time since last follow-up
        - Unresolved medical points"""),
        HumanMessage(content=f"""Last Follow-up: {state['last_follow_up']}
        Conversation History:\n{state['conversation_history']}""")
    ]
    
    questions = analysis_llm.invoke(messages).content
    return {
        "conversation_history": [f"Follow-up: {questions}"],
        "last_follow_up": datetime.datetime.now(),
        "next_action": "supervisor"
    }

# Build workflow
workflow = StateGraph(AgentState)
nodes = {
    "supervisor": supervisor_node,
    "collect_symptoms": handle_symptoms,
    "process_report": process_test_report,
    "clarify_questions": clarify_questions,
    "follow_up": follow_up
}

for name, node in nodes.items():
    workflow.add_node(name, node)

workflow.add_conditional_edges(
    "supervisor",
    lambda state: state["next_action"],
    {action: action for action in nodes.keys() if action != "supervisor"} | {"exit": END}
)

for node in ["collect_symptoms", "process_report", "clarify_questions", "follow_up"]:
    workflow.add_edge(node, "supervisor")

workflow.set_entry_point("supervisor")
agent = workflow.compile()


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