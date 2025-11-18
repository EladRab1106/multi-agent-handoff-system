from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional


class AgentName(str, Enum):
    SUPERVISOR = "supervisor"
    RESEARCHER = "researcher"
    DOCUMENT_CREATOR = "document_creator"


@dataclass
class HandoffMessage:
    """Structured message passed between agents in the workflow.

    Fields
    ------
    task_name: str
        Logical name of the current task (e.g., "research_company", "company_research", "create_report").
    payload: Dict[str, Any] | Any
        Task-specific data. For this project it is either a company name payload
        or structured research data.
    next_agent: Optional[AgentName]
        Which agent should handle the message next. None when returning to the user.
    status: str
        e.g., "pending", "in_progress", "completed", "failed".
    file_path: Optional[str]
        Set by the document creator once a report file has been written.
    error: Optional[str]
        Any error message encountered by an agent.
    """

    task_name: str
    payload: Any
    next_agent: Optional[AgentName]
    status: str
    file_path: Optional[str] = None
    error: Optional[str] = None


__all__ = ["HandoffMessage", "AgentName"]
