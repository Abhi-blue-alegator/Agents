from typing import TypedDict, List, Dict, Optional
from langchain_community.document_loaders import Docx2txtLoader
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_models import ChatOpenAI
import re

# ==================
# 1. Enhanced State
# ==================
class AgentState(TypedDict):
    messages: List[Dict[str, str]]
    symptoms_summary: Optional[str]
    report_text: Optional[str]
    verification_questions: List[str]
    verification_answers: Dict[str, str]
    follow_up_tasks: List[str]
    uploaded_files: List[str]
    conversation_phase: str  # Track phase: "symptoms", "report", "verification"

# ==================
# 2. Improved Nodes
# ==================
def symptom_collector_node(state: AgentState):
    if state.get("conversation_phase") != "symptoms":
        return state
        
    memory = ConversationBufferMemory()
    for msg in state["messages"]:
        if msg["type"] == "human":
            memory.chat_memory.add_user_message(msg["content"])
        else:
            memory.chat_memory.add_ai_message(msg["content"])

    # Detect if summary already exists
    if state.get("symptoms_summary"):
        return state

    # Structured interview with exit condition
    prompt = ChatPromptTemplate.from_messages([
        ("system", """Conduct medical interview. Ask ONE question per response.
        When finished, write: 'SUMMARY: [structured summary]'""")
    ])
    
    chain = prompt | llm
    response = chain.invoke({"input": memory.load_memory_variables({})})
    
    # Extract summary if present
    if "SUMMARY:" in response.content:
        summary = re.search(r"SUMMARY: (.*)", response.content, re.DOTALL).group(1)
        state["symptoms_summary"] = summary.strip()
        state["conversation_phase"] = "report"
    else:
        state["messages"].append({"type": "ai", "content": response.content})
    
    return state

def test_report_processor_node(state: AgentState):
    if state["conversation_phase"] != "report" or not state.get("uploaded_files"):
        return state

    # Load document only once
    if not state.get("report_text"):
        loader = Docx2txtLoader(state["uploaded_files"][0])
        docs = loader.load()
        state["report_text"] = "\n".join([doc.page_content for doc in docs])
    
    # Process only report-related questions
    last_msg = state["messages"][-1]
    if last_msg["type"] == "human" and "report" in last_msg["content"].lower():
        qa_prompt = ChatPromptTemplate.from_template("""
        Report: {report}
        Question: {question}
        Answer:""")
        
        answer = qa_prompt | llm
        response = answer.invoke({
            "report": state["report_text"],
            "question": last_msg["content"]
        })
        state["messages"].append({"type": "ai", "content": response.content})
    
    return state

def verification_generator_node(state: AgentState):
    if state["conversation_phase"] != "report" or not state.get("report_text"):
        return state

    if not state.get("verification_questions"):
        prompt = ChatPromptTemplate.from_template("""
        Generate 3 verification questions for these findings:
        {report}
        Format as numbered questions.""")
        
        questions = prompt | llm
        response = questions.invoke({"report": state["report_text"]})
        state["verification_questions"] = [
            q.strip() for q in response.content.split("\n") if q.strip().startswith("1") or q.strip().startswith("2") or q.strip().startswith("3")
        ][:3]
        state["conversation_phase"] = "verification"
        state["messages"].append({"type": "ai", "content": "Please verify:\n" + "\n".join(state["verification_questions"])})
    
    return state

# ==================
# 3. Enhanced Workflow
# ==================
workflow = StateGraph(AgentState)

workflow.add_node("symptoms", symptom_collector_node)
workflow.add_node("report", test_report_processor_node)
workflow.add_node("verification", verification_generator_node)
workflow.add_node("reporting", follow_up_reporter_node)

workflow.set_entry_point("symptoms")

# Conditionals
workflow.add_conditional_edges(
    "symptoms",
    lambda state: "symptoms" if not state.get("symptoms_summary") else "report",
    {"symptoms": "symptoms", "report": "report"}
)

workflow.add_conditional_edges(
    "report",
    lambda state: "verification" if state.get("report_text") else "reporting",
    {"verification": "verification", "reporting": "reporting"}
)

workflow.add_edge("verification", "reporting")
workflow.add_edge("reporting", END)

# ==================
# 4. Execution Test
# ==================
if __name__ == "__main__":
    initial_state = {
        "messages": [{"type": "human", "content": "I'm having chest pain"}],
        "symptoms_summary": None,
        "report_text": None,
        "verification_questions": [],
        "verification_answers": {},
        "follow_up_tasks": [],
        "uploaded_files": ["report.docx"],
        "conversation_phase": "symptoms"
    }
    
    for step in medical_agent.stream(initial_state):
        node, new_state = next(iter(step.items()))
        print(f"=== {node} ===")
        print(new_state["messages"][-1]["content"])
        print("\n---\n")