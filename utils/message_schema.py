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
    task_name: str
    payload: Any
    next_agent: Optional[AgentName]
    status: str
    file_path: Optional[str] = None
    error: Optional[str] = None


__all__ = ["HandoffMessage", "AgentName"]
