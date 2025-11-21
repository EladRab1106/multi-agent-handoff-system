from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Tuple

from urllib.parse import urlparse

from tavily import TavilyClient
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

from utils.message_schema import HandoffMessage, AgentName


TAVILY_ENDPOINT = "https://api.tavily.com/search"


def _tavily_search(query: str, max_results: int = 5) -> Dict[str, Any]:
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        raise RuntimeError("Tavily API key not set")

    try:
        client = TavilyClient(api_key=api_key)
        data: Dict[str, Any] = client.search(query=query, max_results=max_results)
        return data
    except Exception as exc:
        raise RuntimeError(f"Tavily client error: {exc}") from exc


def _build_queries(company_name: str) -> Dict[str, str]:
    base = company_name
    return {
        "overview": f"{base} company overview background",
        "products": f"{base} products services offerings list",
        "financials": f"{base} financial results revenue profit growth",
        "competitors": f"{base} main competitors market analysis alternatives",
        "news": f"Latest news about {base}",
    }


def _run_all_sections(company_name: str) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, List[str]]]:
    queries = _build_queries(company_name)
    results_by_section: Dict[str, List[Dict[str, Any]]] = {}
    answers_by_section: Dict[str, List[str]] = {}

    for section, query in queries.items():
        section_results: List[Dict[str, Any]] = []
        section_answers: List[str] = []
        try:
            data = _tavily_search(query)
        except Exception as exc:
            raise RuntimeError(f"Tavily search failed: {exc}")

        answer = data.get("answer") or ""
        if isinstance(answer, str) and answer.strip():
            section_answers.append(answer.strip())

        results = data.get("results") or []
        for r in results:
            if isinstance(r, dict):
                section_results.append(r)

        results_by_section[section] = section_results
        answers_by_section[section] = section_answers

    return results_by_section, answers_by_section


def _collect_text(results: List[Dict[str, Any]], answers: List[str]) -> str:
    parts: List[str] = []
    for a in answers:
        if a:
            parts.append(a)
    for r in results:
        content = r.get("content") or ""
        if content:
            parts.append(content)
    return "\n\n".join(parts)


class ResearcherAgent:

    def __init__(self, model: ChatOpenAI) -> None:
        self.model = model

        system_prompt = (
            "You are a company research extraction engine. "
            "You are given pre-fetched Tavily search texts for a company. "
            "Using ONLY the provided texts and URLs, you must produce strict JSON with keys: "
            "company, summary, products, financials, competitors.\n"
            "- 'summary': 13 paragraphs summarizing the company and its main business, if possible.\n"
            "- 'products': list of key product or solution names (strings). If you cannot identify any, use an empty list [].\n"
            "- 'financials': short prose summarizing revenue/profit/financial highlights. If nothing concrete is found, use an empty string ''.\n"
            "- 'competitors': list of competitor company names (strings). If you cannot identify any, use an empty list [].\n"
            "The 'sources' field will be populated programmatically from the Tavily results; do NOT attempt to create or modify it in the JSON.\n"
            "Do NOT use boilerplate placeholders like 'No information found', 'Data not available', or 'Details were limited'. "
            "When information is sparse, return '' (empty string) or [] (empty list) for that field instead. "
            "Return ONLY valid JSON and nothing else."
        )

        self._extraction_chain = (
            ChatPromptTemplate.from_messages(
                [
                    ("system", system_prompt),
                    (
                        "human",
                        "Company name: {company_name}\n\n"
                        "Overview texts:\n{overview_text}\n\n"
                        "Products texts:\n{products_text}\n\n"
                        "Financials texts:\n{financials_text}\n\n"
                        "Competitors texts:\n{competitors_text}\n\n"
                        "News texts:\n{news_text}\n\n"
                        "Candidate source URLs (one per line):\n{sources_urls}\n\n"
                        "Identify competitors based on the inferred industry. If the company is a tech company, list tech competitors. "
                        "If it is a cybersecurity company, list cybersecurity competitors. Only use the content provided.\n\n"
                        "Produce the strict JSON now.",
                    ),
                ]
            )
            | self.model
            | StrOutputParser()
        )

    def run(self, message: HandoffMessage) -> HandoffMessage:
        if message.next_agent != AgentName.RESEARCHER:
            return HandoffMessage(
                task_name=message.task_name,
                payload=message.payload,
                next_agent=AgentName.SUPERVISOR,
                status="failed",
                error="Researcher received message not addressed to it.",
            )

        if message.task_name != "company_research":
            return HandoffMessage(
                task_name=message.task_name,
                payload=message.payload,
                next_agent=AgentName.SUPERVISOR,
                status="failed",
                error=f"Researcher cannot handle task_name={message.task_name}",
            )

        company_name = message.payload.get("company_name") if isinstance(message.payload, dict) else None
        if not company_name:
            return HandoffMessage(
                task_name=message.task_name,
                payload=message.payload,
                next_agent=AgentName.SUPERVISOR,
                status="failed",
                error="Researcher requires 'company_name' in payload.",
            )

        try:
            results_by_section, answers_by_section = _run_all_sections(company_name)

            overview_text = _collect_text(results_by_section.get("overview", []), answers_by_section.get("overview", []))
            products_text = _collect_text(results_by_section.get("products", []), answers_by_section.get("products", []))
            financials_text = _collect_text(results_by_section.get("financials", []), answers_by_section.get("financials", []))
            competitors_text = _collect_text(results_by_section.get("competitors", []), answers_by_section.get("competitors", []))
            news_text = _collect_text(results_by_section.get("news", []), answers_by_section.get("news", []))
            urls_seen: set[str] = set()
            sources_urls: List[str] = []
            for section_results in results_by_section.values():
                for r in section_results:
                    if not isinstance(r, dict):
                        continue
                    source_obj = r.get("source") or {}
                    raw_url = (
                        r.get("url")
                        or r.get("link")
                        or source_obj.get("url")
                        or source_obj.get("id")
                        or ""
                    )
                    url = str(raw_url).strip()
                    if not url or url in urls_seen:
                        continue
                    urls_seen.add(url)
                    sources_urls.append(url)

            sources_urls_str = "\n".join(sources_urls)
            raw_json = self._extraction_chain.invoke(
                {
                    "company_name": company_name,
                    "overview_text": overview_text,
                    "products_text": products_text,
                    "financials_text": financials_text,
                    "competitors_text": competitors_text,
                    "news_text": news_text,
                    "sources_urls": sources_urls_str,
                }
            )

            try:
                structured: Dict[str, Any] = json.loads(raw_json)
            except json.JSONDecodeError:
                start = raw_json.find("{")
                end = raw_json.rfind("}") + 1
                structured = json.loads(raw_json[start:end])

            allowed = {"company", "summary", "products", "financials", "competitors", "sources"}
            structured = {k: v for k, v in structured.items() if k in allowed}

            structured.setdefault("company", company_name)
            structured.setdefault("summary", "")
            structured.setdefault("products", [])
            structured.setdefault("financials", "")
            structured.setdefault("competitors", [])
            structured.setdefault("sources", [])

            structured["sources"] = sources_urls

            return HandoffMessage(
                task_name=message.task_name,
                payload=structured,
                next_agent=AgentName.SUPERVISOR,
                status="completed",
            )

        except Exception as exc:
            return HandoffMessage(
                task_name=message.task_name,
                payload=message.payload,
                next_agent=AgentName.SUPERVISOR,
                status="failed",
                error=str(exc),
            )
