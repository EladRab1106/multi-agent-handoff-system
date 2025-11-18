from dotenv import load_dotenv
load_dotenv()

import os
from typing import Optional

from agents.supervisor import SupervisorAgent
from agents.researcher import ResearcherAgent
from agents.document_creator import DocumentCreatorAgent
from utils.message_schema import HandoffMessage, AgentName

from langchain_openai import ChatOpenAI


def build_model() -> ChatOpenAI:
    """Instantiate the shared LLM for all agents.

    Uses environment variables configured via .env (e.g., OPENAI_API_KEY).
    """
    model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    temperature = float(os.getenv("OPENAI_TEMPERATURE", "0.2"))
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set.")
    return ChatOpenAI(model=model_name, temperature=temperature, api_key=api_key)


def run_supervisor(company_name: str, model: Optional[ChatOpenAI] = None) -> HandoffMessage:
    """Run the full supervisor-managed workflow for a given company name."""
    if not company_name or not company_name.strip():
        raise ValueError("company_name must be a non-empty string")

    if model is None:
        model = build_model()

    researcher = ResearcherAgent(model=model)
    document_creator = DocumentCreatorAgent(model=model)
    supervisor = SupervisorAgent(
        model=model,
        researcher=researcher,
        document_creator=document_creator,
    )

    initial_message = HandoffMessage(
        task_name="research_company",
        payload={"company_name": company_name.strip()},
        next_agent=AgentName.SUPERVISOR,
        status="pending",
    )

    return supervisor.run(initial_message)


def main() -> None:
    """CLI entrypoint.

    Example interaction:
      $ python main.py
      Enter a company name to research: Check Point
      ...
    """
    

    company_name = input("Enter a company name to research: ").strip()
    final_message = run_supervisor(company_name)

    if final_message.status != "completed":
        print("Workflow did not complete successfully.")
        if final_message.error:
            print(f"Error: {final_message.error}")
        return

    print("Research report successfully created!")
    if final_message.file_path:
        print(f"Report file: {final_message.file_path}")


if __name__ == "__main__":
    main()