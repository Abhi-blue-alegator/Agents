from autogen import AssistantAgent, UserProxyAgent, GroupChatManager, GroupChat
from langchain_community.document_loaders import Docx2txtLoader
from typing import Dict, List, Optional
import os

# Configuration
config_list = [{"model": "gpt-4", "api_key": os.getenv("OPENAI_API_KEY")}]

class MedicalAgentSystem:
    def __init__(self):
        # Initialize core agents
        self.user_proxy = UserProxyAgent(
            name="Patient_Proxy",
            human_input_mode="ALWAYS",
            code_execution_config=False,
            max_consecutive_auto_reply=1,
            is_termination_msg=lambda x: "TERMINATE" in x.get("content", "")
        )
        
        # Phase 1: Symptom Collection Agent
        self.symptom_agent = AssistantAgent(
            name="Symptom_Collector",
            system_message="""You are a medical interviewer. Ask ONE question at a time about:
                            - Symptom description
                            - Duration
                            - Severity (1-10)
                            - other relevant questions
                            ask atleast 5 questions
                            End with 'SUMMARY: [summary]' when complete.""",
            llm_config={"config_list": config_list}
        )
        
        # Phase 2: Report Analysis Agent
        self.report_agent = AssistantAgent(
            name="Report_Analyzer",
            system_message="""Analyze medical reports. Respond ONLY to report-related queries.
                            Format:
                            - Summary: 2-3 sentences
                            - Key Findings: Bullet points
                            - Recommendations: Clinical steps
                            Say "I don't see any reports" if none provided.""",
            llm_config={"config_list": config_list}
        )
        
        # Phase 3: Verification Agent
        self.verification_agent = AssistantAgent(
            name="Symptom_Verifier",
            system_message="""ask the patient if he/she is experiencing the symptoms that are tested positive in the report. atleast ask 5 questions. Ask ONE question at a time.
                            Format: "VERIFICATION: [question]" """,
            llm_config={"config_list": config_list}
        )
        
        # Phase 4: Reporting Agent
        self.doctor_liaison = AssistantAgent(
            name="Doctor_Liaison",
            system_message="""Generate final reports ONLY when asked. Include:
                            - Patient Summary
                            - Test Correlation
                            - Urgency Level
                            - Next Steps""",
            llm_config={"config_list": config_list}
        )

        # Phase-specific group chats
        self.symptom_chat = GroupChat(
            agents=[self.user_proxy, self.symptom_agent],
            messages=[],
            max_round=40,
            speaker_selection_method="round_robin"
        )
        
        self.report_chat = GroupChat(
            agents=[self.user_proxy, self.report_agent],
            messages=[],
            max_round=40,
            speaker_selection_method="round_robin"
        )
        
        self.verification_chat = GroupChat(
            agents=[self.user_proxy, self.verification_agent],
            messages=[],
            max_round=40,
            speaker_selection_method="round_robin"
        )

    def run_interview(self):
        print("="*40)
        print(" Medical Interview Session Started ")
        print("="*40)
        
        # Phase 1: Symptom Collection
        symptom_manager = GroupChatManager(groupchat=self.symptom_chat)
        self.user_proxy.initiate_chat(
            symptom_manager,
            message="Let's begin the symptom assessment."
        )
        
        # Get symptom summary
        symptom_summary = self._extract_summary(self.symptom_chat.messages)
        
        # Phase 2: Report Handling
        doc_path = input("\nUpload DOCX report path (or Enter to skip): ").strip()
        report_text = ""
        if doc_path:
            report_text = self.process_document(doc_path)
            report_manager = GroupChatManager(groupchat=self.report_chat)
            self.user_proxy.initiate_chat(
                report_manager,
                message=f"Analyze this report:\n{report_text}"
            )
        
        # Phase 3: Verification
        if report_text:
            verification_manager = GroupChatManager(groupchat=self.verification_chat)
            self.user_proxy.initiate_chat(
                verification_manager,
                message=f"Verify symptoms:\n{symptom_summary}\nReport:{report_text}"
            )
        
        # Phase 4: Final Report
        final_report = self.doctor_liaison.generate_reply(
            messages=[{
                "content": f"Generate report:\nSymptoms:{symptom_summary}\nReport:{report_text}",
                "role": "user"
            }]
        )
        print("\n" + "="*40)
        print(final_report)
        print("="*40)

    def process_document(self, path: str) -> str:
        try:
            loader = Docx2txtLoader(path)
            return "\n".join([doc.page_content for doc in loader.load()])
        except Exception as e:
            return f"Error processing document: {str(e)}"

    def _extract_summary(self, messages: List[dict]) -> str:
        for msg in reversed(messages):
            if "SUMMARY:" in msg["content"]:
                return msg["content"].split("SUMMARY:")[-1].strip()
        return "No summary found"

if __name__ == "__main__":
    MedicalAgentSystem().run_interview()