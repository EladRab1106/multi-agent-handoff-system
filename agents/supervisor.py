from __future__ import annotations

from dataclasses import asdict
from typing import Any

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

from utils.message_schema import HandoffMessage, AgentName


class SupervisorAgent:
    """Supervisor that orchestrates the strict handoff workflow.

    Workflow (fixed order):
      User -> Supervisor -> Researcher -> Supervisor -> DocumentCreator -> Supervisor -> User
    """

    def __init__(
        self,
        model: ChatOpenAI,
        researcher: Any,
        document_creator: Any,
    ) -> None:
        self.model = model
        self.researcher = researcher
        self.document_creator = document_creator

        # Simple chain used when the supervisor needs to reason about the task
        system_prompt = (
            "You are a supervisor agent coordinating a research workflow. "
            "Your job is to understand the user request and pass clear, concise "
            "instructions to specialized agents. Do NOT perform their tasks yourself."
        )
        self._thinking_chain = (
            ChatPromptTemplate.from_messages(
                [
                    ("system", system_prompt),
                    (
                        "human",
                        "User request: {user_request}\n\n"
                        "Summarize the request as a short research instruction for a researcher agent.",
                    ),
                ]
            )
            | self.model
            | StrOutputParser()
        )

    def _interpret_request(self, user_request: str) -> str:
        """Use the LLM to produce a concise research instruction."""
        return self._thinking_chain.invoke({"user_request": user_request}).strip()

    def run(self, message: HandoffMessage) -> HandoffMessage:
        """Run the full workflow from the supervisor's perspective.

        This method assumes it is called first with next_agent=SUPERVISOR and
        task_name="research_company".
        """
        try:
            if message.next_agent != AgentName.SUPERVISOR:
                raise ValueError("Supervisor.run must be entered with next_agent=SUPERVISOR")

            if message.task_name != "research_company":
                raise ValueError(f"Unsupported task_name for supervisor: {message.task_name}")

            company_name = message.payload.get("company_name") if isinstance(message.payload, dict) else None
            if not company_name:
                raise ValueError("Supervisor requires 'company_name' in payload to start workflow")

            # Step 1: Understand the task (but still pass only the company name downstream)
            _ = self._interpret_request(f"Research the company {company_name}")

            # Step 2: Handoff to Researcher
            to_researcher = HandoffMessage(
                task_name="company_research",
                payload={"company_name": company_name},
                next_agent=AgentName.RESEARCHER,
                status="in_progress",
            )

            researcher_result = self.researcher.run(to_researcher)
            if researcher_result.status != "completed":
                return HandoffMessage(
                    task_name=message.task_name,
                    payload=asdict(researcher_result),
                    next_agent=AgentName.SUPERVISOR,
                    status="failed",
                    error=researcher_result.error or "Researcher did not complete successfully",
                )

            # Step 3: Handoff to Document Creator with structured research
            research_payload = researcher_result.payload
            to_doc = HandoffMessage(
                task_name="create_report",
                payload={
                    "company_name": company_name,
                    "research": research_payload,
                },
                next_agent=AgentName.DOCUMENT_CREATOR,
                status="in_progress",
            )

            doc_result = self.document_creator.run(to_doc)
            if doc_result.status != "completed" or not doc_result.file_path:
                return HandoffMessage(
                    task_name=message.task_name,
                    payload=asdict(doc_result),
                    next_agent=AgentName.SUPERVISOR,
                    status="failed",
                    error=doc_result.error or "Document creator did not produce a file",
                )

            # Step 4: Final response back to user
            return HandoffMessage(
                task_name=message.task_name,
                payload={
                    "message": f"Research report for {company_name} created successfully.",
                    "research": research_payload,
                },
                next_agent=None,
                status="completed",
                file_path=doc_result.file_path,
            )

        except Exception as exc:  # noqa: BLE001
            return HandoffMessage(
                task_name=message.task_name,
                payload=asdict(message),
                next_agent=AgentName.SUPERVISOR,
                status="failed",
                error=str(exc),
            )