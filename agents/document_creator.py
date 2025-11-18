from __future__ import annotations

from typing import Any, Dict

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

from utils.file_writer import write_report_file
from utils.message_schema import HandoffMessage, AgentName


class DocumentCreatorAgent:
    """Agent that converts research JSON into a formatted report file.

    It receives only structured research data and is solely responsible for
    file creation and formatting.
    """

    def __init__(self, model: ChatOpenAI) -> None:
        self.model = model

        system_prompt = (
            "You are a document creation assistant. "
            "Given structured research data about a company, craft a clear, "
            "well-organized markdown report. Use headings, bullet points, and short paragraphs. "
            "You must ONLY use the information present in the JSON. Do NOT fabricate products, "
            "numbers, or boilerplate text like 'No detailed summary available', 'Not available', "
            "or 'No products listed'. If a field's value is an empty string or an empty list, "
            "you may omit that section instead of filling it with placeholder prose. "
            "If the JSON includes a non-empty 'sources' list of URL strings, add a '## Sources' section and render "
            "each source URL as a markdown bullet in the form '- URL'. "
            "If the 'sources' list is empty or missing, still include a '## Sources' section containing exactly the "
            "sentence 'No sources were provided.'. Do not invent, rewrite, or paraphrase this sentence. Never write "
            "the phrase 'Details were limited in the research results.' anywhere in the report."
        )

        self._formatting_chain = (
            ChatPromptTemplate.from_messages(
                [
                    ("system", system_prompt),
                    (
                        "human",
                        "Company name: {company_name}\n\n"
                        "Structured research JSON:\n{research_json}\n\n"
                        "Write the full markdown report now.",
                    ),
                ]
            )
            | self.model
            | StrOutputParser()
        )

    def run(self, message: HandoffMessage) -> HandoffMessage:
        if message.next_agent != AgentName.DOCUMENT_CREATOR:
            return HandoffMessage(
                task_name=message.task_name,
                payload=message.payload,
                next_agent=AgentName.SUPERVISOR,
                status="failed",
                error="DocumentCreator received message not addressed to it.",
            )

        if message.task_name != "create_report":
            return HandoffMessage(
                task_name=message.task_name,
                payload=message.payload,
                next_agent=AgentName.SUPERVISOR,
                status="failed",
                error=f"DocumentCreator cannot handle task_name={message.task_name}",
            )

        payload = message.payload if isinstance(message.payload, dict) else {}
        company_name = payload.get("company_name")
        research: Dict[str, Any] | None = payload.get("research")
        if not company_name or research is None:
            return HandoffMessage(
                task_name=message.task_name,
                payload=message.payload,
                next_agent=AgentName.SUPERVISOR,
                status="failed",
                error="DocumentCreator requires 'company_name' and 'research' in payload.",
            )

        try:
            import json

            research_json = json.dumps(research, indent=2, ensure_ascii=False)
            markdown_report = self._formatting_chain.invoke(
                {"company_name": company_name, "research_json": research_json}
            )

            file_path = write_report_file(content=markdown_report, company_name=company_name, fmt="markdown")

            return HandoffMessage(
                task_name=message.task_name,
                payload={"message": f"Report created for {company_name}."},
                next_agent=AgentName.SUPERVISOR,
                status="completed",
                file_path=file_path,
            )

        except Exception as exc:  # noqa: BLE001
            return HandoffMessage(
                task_name=message.task_name,
                payload=message.payload,
                next_agent=AgentName.SUPERVISOR,
                status="failed",
                error=str(exc),
            )