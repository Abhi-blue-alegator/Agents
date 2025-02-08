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
    next_action: Literal[
        "collect_symptoms", 
        "process_report",
        "clarify_questions",
        "follow_up",
        "exit"
    ]
    user_input: str

# Initialize LLMs
supervisor_llm = ChatOpenAI(temperature=0.1, model="gpt-4-turbo")
symptom_llm = ChatOpenAI(temperature=0.2, model="gpt-3.5-turbo")
analysis_llm = ChatOpenAI(temperature=0.1, model="gpt-4o")
summary_llm = ChatOpenAI(temperature=0.1, model="gpt-4-turbo")

def supervisor_node(state: AgentState):
    messages = [
        SystemMessage(content="""You are a medical workflow supervisor. Decide next action based on:
        1. Continue symptom collection until at least 5 patient responses
        2. Process test reports immediately when uploaded
        3. Address clarification questions before follow-ups
        4. Only exit when symptoms are collected and reports processed
        
        Current State:
        Symptoms Collected: {symptoms_collected}
        Test Report: {test_report_status}
        Pending Questions: {pending_questions}
        Conversation Length: {conv_len}""".format(
            symptoms_collected=state["symptoms_collected"],
            test_report_status="Uploaded" if state["test_report"] else "None",
            pending_questions=len(state["pending_questions"]),
            conv_len=len(state["conversation_history"])
        )),
        HumanMessage(content="Last 3 messages:\n" + "\n".join(state["conversation_history"][-3:]))
    ]
    
    decision = supervisor_llm.invoke(messages).content.lower().strip()
    return {"next_action": "collect_symptoms" if decision == "exit" and len(state["conversation_history"]) < 5 else decision}

def handle_symptoms(state: AgentState):
    messages = [
        SystemMessage(content="""You are a persistent medical assistant. Even if patient is brief:
        1. Ask specific symptom questions
        2. Request details about duration, intensity, location
        3. Ask one question at a time
        4. Maintain friendly tone"""),
        HumanMessage(content=f"Conversation History:\n{state['conversation_history']}\nPatient Input: {state['user_input']}")
    ]
    
    response = symptom_llm.invoke(messages).content
    new_state = {
        "conversation_history": [
            f"Patient: {state['user_input']}",
            f"Assistant: {response}"
        ],
        "next_action": "supervisor"
    }
    
    if len(state["conversation_history"]) > 6:
        new_state["symptoms_collected"] = True
    return new_state

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

# Modified chat interface
def chat_interface():
    state = {
        "conversation_history": [],
        "test_report": "",
        "generated_questions": [],
        "pending_questions": [],
        "last_follow_up": None,
        "symptoms_collected": False,
        "next_action": "collect_symptoms",
        "user_input": ""
    }
    
    print("Medical Assistant: Hello! I'm your health assistant. Let's start with your symptoms.")
    
    while True:
        if state["next_action"] == "process_report":
            report_path = input("\n[System] Please upload test report path: ")
            state["test_report"] = report_path
            state["user_input"] = "[REPORT_UPLOADED]"
        else:
            user_input = input("\nPatient: ")
            state["user_input"] = user_input
        
        result = agent.invoke(state)
        state.update(result)
        
        # Print latest assistant message
        if state["conversation_history"]:
            print(f"\nAssistant: {state['conversation_history'][-1].split('Assistant: ')[-1]}")
        
        if state.get("next_action") == "exit":
            summary = generate_summary(state)
            print(f"\nConsultation Summary for Doctor:\n{summary}")
            break

# Keep workflow configuration and other nodes

if __name__ == "__main__":
    chat_interface()