from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Literal

from docx import Document


ROOT_DIR = Path(__file__).resolve().parents[1]
OUTPUTS_DIR = ROOT_DIR / "outputs"


def _ensure_outputs_dir() -> None:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


def _slugify(value: str) -> str:
    return "".join(c.lower() if c.isalnum() else "-" for c in value).strip("-") or "report"


def write_report_file(
    content: str,
    company_name: str,
    fmt: Literal["markdown", "docx"] = "markdown",
) -> str:
    _ensure_outputs_dir()

    timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    base_name = f"{_slugify(company_name)}-{timestamp}"

    if fmt == "markdown":
        file_path = OUTPUTS_DIR / f"{base_name}.md"
        file_path.write_text(content, encoding="utf-8")
    elif fmt == "docx":
        file_path = OUTPUTS_DIR / f"{base_name}.docx"
        doc = Document()
        for line in content.splitlines():
            doc.add_paragraph(line)
        doc.save(file_path)
    else:
        raise ValueError(f"Unsupported report format: {fmt}")

    return str(file_path)
