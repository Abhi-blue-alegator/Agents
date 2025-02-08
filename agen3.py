from autogen import AssistantAgent, UserProxyAgent, GroupChatManager, GroupChat
from langchain_community.document_loaders import Docx2txtLoader
from typing import Dict, List, Optional
import os
from datetime import datetime

# Configuration
config_list = [{"model": "gpt-4o-mini", "api_key": os.getenv("OPENAI_API_KEY")}]

# ==================
# 1. Medical Agent System
# ==================
class MedicalAgentSystem:
    def __init__(self):
        # Initialize agents
        self.user_proxy = UserProxyAgent(
            name="Patient_Proxy",
            human_input_mode="ALWAYS",
            code_execution_config=False,
            max_consecutive_auto_reply=0
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
            system_message="""Analyze medical reports directly provided as text. Always respond with:
                            - Report Summary: 2-3 sentence overview
                            - Key Findings: Bullet points
                            - Recommendations: Clinical suggestions
                            Never mention inability to process documents.""",
            llm_config={"config_list": config_list}
        )
        
        self.verification_agent = AssistantAgent(
            name="Symptom_Verifier",
            system_message="""Generate verification questions based on test reports,
                             after the test_report is analyzed. Ask ONE question at a time.End with 'SUMMARY: [summary]' 
                            when complete.""",
            llm_config={"config_list": config_list}
        )
        
        self.doctor_liaison = AssistantAgent(
            name="Doctor_Liaison",
            system_message="""Generate comprehensive reports for the doctor to read in the end with:
                            - Patient Summary
                            - Test Correlation
                            - Urgency Assessment
                            do not try to be a replacement for the doctor.""",
            llm_config={"config_list": config_list}
        )

        # Configure group chat
        self.group_chat = GroupChat(
            agents=[self.user_proxy, self.symptom_agent, self.report_agent, 
                   self.verification_agent, self.doctor_liaison],
            messages=[],
            max_round=40
        )
        self.manager = GroupChatManager(
            groupchat=self.group_chat, 
            llm_config={"config_list": config_list}
        )
        
        # Initialize data stores
        self.verification_data = {}
        self.report_text = ""

    # ==================
    # 2. Core Functionality
    # ==================
    def process_document(self, file_path: str) -> str:
        """Process DOCX files using LangChain loader"""
        try:
            loader = Docx2txtLoader(file_path)
            docs = loader.load()
            return "\n\n".join([doc.page_content for doc in docs if doc.page_content.strip()])
        except Exception as e:
            return f"Document processing error: {str(e)}"

    def analyze_report(self, doc_text: str) -> str:
        """Force structured report analysis"""
        prompt = f"""
        MEDICAL REPORT ANALYSIS TASK:
        {doc_text}
        
        Respond EXACTLY in this format:
        
        Report Summary: [2-3 sentence overview]
        
        Key Findings:
        - Finding 1
        - Finding 2
        - Finding 3
        
        Recommendations:
        - Recommendation 1
        - Recommendation 2
        """
        return self.report_agent.generate_reply(
            messages=[{"role": "user", "content": prompt}]
        )

    def generate_verification_questions(self) -> List[str]:
        """Create symptom verification questions"""
        prompt = f"""
        Symptom Summary: {self.extract_summary()}
        Test Report: {self.report_text}
        
        Generate 3 verification questions to confirm symptom-report correlation.
        Format as:
        1. Question 1
        2. Question 2
        3. Question 3
        """
        response = self.verification_agent.generate_reply(
            messages=[{"role": "user", "content": prompt}]
        )
        return [q.split(" ", 1)[1].strip() for q in response.split("\n")[:3]]

    # ==================
    # 3. Conversation Flow
    # ==================
    def run_interview(self):
        print("\n" + "="*40)
        print(" Medical Interview Session Started ")
        print("="*40 + "\n")
        
        # Phase 1: Symptom Collection
        self.user_proxy.initiate_chat(
            self.manager,
            message="we shall begin the symptom assessment."
        )
        
        # Phase 2: Report Handling
        doc_path = input("\nUpload DOCX report path (or press Enter to skip): ").strip()
        if doc_path:
            self.report_text = self.process_document(doc_path)
            if not self.report_text.startswith("Error"):
                print("\nAnalyzing report...")
                analysis = self.analyze_report(self.report_text)
                print(f"\nReport Analysis:\n{analysis}")
                
                # Phase 3: Verification Questions
                questions = self.generate_verification_questions()
                print("\nVerification Questions:")
                for i, q in enumerate(questions, 1):
                    user_input = input(f"{i}. {q}\nYour answer: ")
                    self.verification_data[q] = user_input
        
        # Phase 4: Final Reporting
        print("\nGenerating final report...")
        final_report = self.generate_final_report()
        print("\n" + "="*40)
        print(" Final Doctor Report ")
        print("="*40)
        print(final_report)
        print("\nFollow-up scheduled in 3 days. Thank you!")

    # ==================
    # 4. Reporting & Utilities
    # ==================
    def extract_summary(self) -> str:
        """Extract symptom summary from chat history"""
        for msg in reversed(self.group_chat.messages):
            if "SUMMARY:" in msg["content"]:
                return msg["content"].split("SUMMARY:")[-1].strip()
        return "No symptom summary available"

    def generate_final_report(self) -> str:
        """Compile comprehensive doctor report"""
        prompt = f"""
        Patient Summary:
        {self.extract_summary()}
        
        Test Findings:
        {self.report_text if self.report_text else 'No reports submitted'}
        
        Verification Answers:
        {self.verification_data}
        
        Create a detailed medical report including:
        - Clinical Assessment
        - Urgency Level (Low/Medium/High)
        do not take the role of the doctor in any case.
        """
        return self.doctor_liaison.generate_reply(
            messages=[{"role": "user", "content": prompt}]
        )

# ==================
# 5. Execution
# ==================
if __name__ == "__main__":
    system = MedicalAgentSystem()
    system.run_interview()