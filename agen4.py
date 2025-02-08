from autogen import AssistantAgent, UserProxyAgent, GroupChatManager, GroupChat
from langchain_community.document_loaders import Docx2txtLoader
from typing import Dict, List, Optional
import os
from datetime import datetime

# Configuration
config_list = [{"model": "gpt-4", "api_key": os.getenv("OPENAI_API_KEY")}]

# ==================
# 1. Medical Agent System (Fixed Version)
# ==================
class MedicalAgentSystem:
    def __init__(self):
        # Initialize agents with proper conversation control
        self.user_proxy = UserProxyAgent(
            name="Patient_Proxy",
            human_input_mode="ALWAYS",
            code_execution_config=False,
            max_consecutive_auto_reply=1,  # Critical fix for input handling
            is_termination_msg=lambda x: "SUMMARY:" in x.get("content", "").upper()
        )
        
        self.symptom_agent = AssistantAgent(
            name="Symptom_Collector",
            system_message="""You are a medical interview specialist. Ask ONE question at a time 
                            about symptoms, duration, and severity. End with 'SUMMARY: [summary]' 
                            when complete. After each question, wait for patient response.""",
            llm_config={"config_list": config_list}
        )
        
        self.report_agent = AssistantAgent(
            name="Report_Analyzer",
            system_message="""Analyze medical reports directly provided as text. Format responses as:
                            - Report Summary: 2-3 sentence overview
                            - Key Findings: Bullet points
                            - Recommendations: Clinical suggestions""",
            llm_config={"config_list": config_list}
        )
        
        self.verification_agent = AssistantAgent(
            name="Symptom_Verifier",
            system_message="""Generate verification questions based on test reports. 
                            Ask ONE question at a time. Wait for patient response.""",
            llm_config={"config_list": config_list}
        )
        
        self.doctor_liaison = AssistantAgent(
            name="Doctor_Liaison",
            system_message="""Generate final reports including:
                            - Patient Summary
                            - Test Correlation
                            - Urgency Assessment
                            - Recommended Next Steps""",
            llm_config={"config_list": config_list}
        )

        # Configure group chat with explicit turn control
        self.group_chat = GroupChat(
            agents=[self.user_proxy, self.symptom_agent, self.report_agent, 
                   self.verification_agent, self.doctor_liaison],
            messages=[],
            max_round=40,
            speaker_selection_method="round_robin"  # Ensures proper turn-taking
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
        """Structured report analysis"""
        prompt = f"""
        MEDICAL REPORT ANALYSIS TASK:
        {doc_text}
        
        Respond in this format:
        
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

    # ==================
    # 3. Fixed Conversation Flow
    # ==================
    def run_interview(self):
        print("\n" + "="*40)
        print(" Medical Interview Session Started ")
        print("="*40 + "\n")
        
        # Phase 1: Symptom Collection
        self.user_proxy.initiate_chat(
            self.manager,
            message="Let's begin the symptom assessment.",
            clear_history=True
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
                self._handle_verification_phase()

        # Phase 4: Final Reporting
        self._generate_final_report()

    def _handle_verification_phase(self):
        """Handle verification questions with user input"""
        questions = [
            "Does the report align with your symptoms?",
            "Have you experienced any symptoms mentioned in the report?",
            "Are there any discrepancies between your symptoms and the report?"
        ]
        
        print("\nVerification Questions:")
        for i, q in enumerate(questions, 1):
            self.user_proxy.send(
                message=f"{i}. {q}",
                recipient=self.manager
            )
            answer = input(f"{i}. {q}\nYour answer: ")
            self.verification_data[q] = answer

    def _generate_final_report(self):
        """Generate and display final report"""
        print("\nGenerating final report...")
        final_report = self.doctor_liaison.generate_reply(
            messages=[{
                "role": "user",
                "content": f"""
                Patient Summary: {self._extract_summary()}
                Test Findings: {self.report_text}
                Verification Answers: {self.verification_data}
                
                Create a clinical report with:
                1. Symptom Analysis
                2. Test Correlation
                3. Urgency Level
                4. Recommended Next Steps
                """
            }]
        )
        print("\n" + "="*40)
        print(" Final Doctor Report ")
        print("="*40)
        print(final_report.content)
        print("\nFollow-up scheduled in 3 days. Thank you!")

    def _extract_summary(self) -> str:
        """Extract symptom summary from chat history"""
        for msg in reversed(self.group_chat.messages):
            if "SUMMARY:" in msg["content"]:
                return msg["content"].split("SUMMARY:")[-1].strip()
        return "No symptom summary available"

# ==================
# 4. Execution
# ==================
if __name__ == "__main__":
    system = MedicalAgentSystem()
    system.run_interview()