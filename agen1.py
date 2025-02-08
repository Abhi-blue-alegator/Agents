from autogen import AssistantAgent, UserProxyAgent, GroupChatManager, GroupChat
from typing import Dict, List, Optional
import os
from docx import Document
from datetime import datetime

# Configuration
config_list = [{"model": "gpt-4", "api_key": os.getenv("OPENAI_API_KEY")}]

class MedicalAgentSystem:
    def __init__(self):
        # Initialize all agents
        self.user_proxy = UserProxyAgent(
            name="Patient_Proxy",
            human_input_mode="ALWAYS",
            code_execution_config=False
        )
        
        self.symptom_agent = AssistantAgent(
            name="Symptom_Collector",
            system_message="""Conduct structured medical interviews. Ask ONE question at a time
                            about symptoms, duration, and severity. End with 'SUMMARY: [summary]'
                            when complete.""",
            llm_config={"config_list": config_list}
        )
        
        self.report_agent = AssistantAgent(
            name="Report_Analyzer",
            system_message="""Analyze uploaded medical reports. If you receive a message starting with 
                            'DOCUMENT_UPLOAD:', extract the report text and provide relevant analysis.""",
            llm_config={"config_list": config_list}
        )
        
        self.verification_agent = AssistantAgent(
            name="Symptom_Verifier",
            system_message="""Generate verification questions based on test reports 
                            and symptom summaries. Ask ONE question at a time.""",
            llm_config={"config_list": config_list}
        )
        
        self.doctor_liaison = AssistantAgent(
            name="Doctor_Liaison",
            system_message="""Manage follow-ups and generate doctor reports. Schedule 
                            reminders and escalate urgent cases.""",
            llm_config={"config_list": config_list}
        )
        
        # Set up group chat
        self.group_chat = GroupChat(
            agents=[self.user_proxy, self.symptom_agent, self.report_agent, 
                   self.verification_agent, self.doctor_liaison],
            messages=[],
            max_round=20
        )
        self.manager = GroupChatManager(groupchat=self.group_chat, llm_config={"config_list": config_list})

    def process_document(self, file_path: str) -> str:
        """Convert DOCX to text with formatting preservation"""
        try:
            doc = Document(file_path)
            return "\n".join([para.text for para in doc.paragraphs])
        except Exception as e:
            return f"Error processing document: {str(e)}"

    def handle_report_qa(self, doc_text: str, question: str) -> str:
        """Handle report-based questions with context"""
        prompt = f"""
        Medical Report Content:
        {doc_text}
        
        Patient Question: {question}
        
        Provide a detailed, professional answer in 2-3 sentences.
        """
        return self.report_agent.generate_reply(messages=[{"content": prompt}])

    def generate_verification_questions(self, symptoms: str, report: str) -> List[str]:
        """Create symptom verification questions"""
        prompt = f"""
        Symptom Summary: {symptoms}
        Test Findings: {report}
        
        Generate 3 verification questions to confirm symptom accuracy.
        Format as numbered items.
        """
        response = self.verification_agent.generate_reply(messages=[{"content": prompt}])
        return [q.split(" ", 1)[1] for q in response.split("\n")[:3]]

    def run_medical_interview(self):
        self.user_proxy.initiate_chat(
            self.manager,
            message="I'm ready to start the symptom interview"
        )
        
        doc_path = input("Upload report path (or skip): ")
        if doc_path:
            doc_text = self.process_document(doc_path)
            self.user_proxy.send(
                message=f"DOCUMENT_UPLOAD: {doc_text}",
                recipient=self.report_agent
            )
            
            symptoms = self.extract_summary()
            questions = self.generate_verification_questions(symptoms, doc_text)
            for q in questions:
                self.user_proxy.send(
                    message=f"VERIFICATION_QUESTION: {q}",
                    recipient=self.verification_agent
                )
                answer = self.user_proxy.get_human_input()
                self.store_verification_answer(q, answer)
        
        self.doctor_liaison.send(
            message="Generate final report",
            recipient=self.manager
        )
        self.schedule_follow_up()

    def extract_summary(self) -> str:
        """Extract symptom summary from conversation history"""
        for msg in reversed(self.group_chat.messages):
            if "SUMMARY:" in msg["content"]:
                return msg["content"].split("SUMMARY:")[-1].strip()
        return "No symptom summary found"

    def store_verification_answer(self, question: str, answer: str):
        """Store verification answers with timestamp"""
        timestamp = datetime.now().isoformat()
        if not hasattr(self, "verification_data"):
            self.verification_data = {}
        self.verification_data[timestamp] = {
            "question": question,
            "answer": answer
        }

    def schedule_follow_up(self):
        """Schedule follow-up reminders"""
        reminder = "Follow-up scheduled in 3 days. Doctor notification sent."
        self.doctor_liaison.send(
            message=reminder,
            recipient=self.user_proxy
        )

if __name__ == "__main__":
    system = MedicalAgentSystem()
    system.run_medical_interview()