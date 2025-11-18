# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Key Commands

### Environment setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

These are the standard setup steps described in `README.md`.

### Running the application

From the project root, after activating the virtualenv:

```bash
python main.py
```

The CLI will prompt for a company name and run the full multi-agent workflow, saving the final markdown report into `outputs/`.

### Environment variables

All LLM and search behavior depends on environment variables (as documented in `README.md`):

- `OPENAI_API_KEY` – required, for the LangChain `ChatOpenAI` model.
- `OPENAI_MODEL` – optional, defaults to `gpt-4o-mini`.
- `OPENAI_TEMPERATURE` – optional.
- `TAVILY_API_KEY` – required for web research via Tavily.

These are typically set via a `.env` file in the project root.

### Tests

The repository does not ship with a test suite, but the code is structured for easy testing (as described in the README’s **Testing Guidance**). If you add tests using `pytest`, the usual command from the project root is:

```bash
pytest
```

## High-Level Architecture

### Overview

This repository implements a minimal, end-to-end **multi-agent handoff system** for company research using **LangChain**. The workflow is strictly ordered and agent responsibilities are clearly separated:

```text
User → Supervisor → Researcher → Supervisor → Document Creator → Supervisor → User
```

Each agent communicates via a shared structured message type, `HandoffMessage`, defined in `utils/message_schema.py`.

### Message schema and agent coordination

- `utils/message_schema.py` defines:
  - `AgentName` – enum with `SUPERVISOR`, `RESEARCHER`, and `DOCUMENT_CREATOR`.
  - `HandoffMessage` – dataclass with fields:
    - `task_name`: logical task identifier (e.g., `"research_company"`, `"company_research"`, `"create_report"`).
    - `payload`: task-specific data (either the company name payload or structured research JSON).
    - `next_agent`: which agent should handle the message next (`AgentName` or `None`).
    - `status`: string status (`"pending"`, `"in_progress"`, `"completed"`, `"failed"`).
    - `file_path`: path to the generated report (set by the Document Creator).
    - `error`: error message if any.
    - `meta`: free-form metadata dict.

The **Supervisor** owns the high-level workflow and is the only component allowed to orchestrate handoffs between agents.

### SupervisorAgent (`agents/supervisor.py`)

- Holds a shared `ChatOpenAI` model instance used for brief reasoning about the user request.
- Uses a small internal chain (`_thinking_chain`) with a system prompt that tells it to understand the user request and generate a concise research instruction **without** doing the research itself.
- `run(...)` enforces the fixed workflow and validates inputs:
  - Expects `next_agent == AgentName.SUPERVISOR` and `task_name == "research_company"` on entry.
  - Extracts `company_name` from `message.payload`.
  - Calls `_interpret_request` to get a concise research instruction (used only for reasoning, not persisted).
  - Constructs a `HandoffMessage` for the **ResearcherAgent** (`task_name="company_research"`, `next_agent=RESEARCHER`, `status="in_progress"`).
  - After the researcher completes, forwards the structured research to the **DocumentCreatorAgent** (`task_name="create_report"`).
  - On successful report creation, returns a final `HandoffMessage` to the caller with `status="completed"`, `next_agent=None`, and `file_path` set.

### ResearcherAgent (`agents/researcher.py`)

**Responsibility:** perform web research about a company via Tavily, then use an LLM **only** to extract and structure the data into strict JSON.

Key elements:

1. **Tavily search integration**
   - `_tavily_search(query: str, max_results: int = 5)` uses the official `TavilyClient` and requires `TAVILY_API_KEY`.
   - `_build_queries(company_name: str)` constructs deterministic, section-specific queries **without assuming a specific industry**:
     - `overview`: `"{company_name} company overview background"`
     - `products`: `"{company_name} products services offerings list"`
     - `financials`: `"{company_name} financial results revenue profit growth"`
     - `competitors`: `"{company_name} main competitors market analysis alternatives"`
     - `news`: `"Latest news about {company_name}"`
   - `_run_all_sections(...)` calls Tavily for each section, storing both `answer` strings and `results` lists with debug logging.

2. **Text collection and source URLs**
   - `_collect_text(...)` merges Tavily `answer` fields and `content` fields from results into a section-specific text blob.
   - `run(...)` builds five blobs: `overview_text`, `products_text`, `financials_text`, `competitors_text`, `news_text`.
   - It then builds a canonical list of **source URLs** by scanning Tavily `results` and normalizing fields (`url`, `link`, or nested `source.url`/`source.id`).
   - The URLs are deduplicated and joined into a newline-separated string `sources_urls` for the prompt.

3. **LLM extraction chain**
   - A `ChatPromptTemplate` defines:
     - **System message**: describes the agent as a "company research extraction engine", clarifies that only provided texts and URLs may be used, and requires strict JSON with keys:
       - `company`, `summary`, `products`, `financials`, `competitors`.
       - `summary`: 1–3 paragraphs summarizing the company and its main business, if possible.
       - `products`: list of key product or solution names; must be `[]` if none can be identified.
       - `financials`: short prose summarizing revenue/profit/financial highlights; empty string `""` if none are found.
       - `competitors`: list of competitor company names; `[]` if none can be identified.
       - `sources` is **never** generated by the LLM and is populated programmatically instead.
       - The prompt forbids boilerplate placeholder strings like "No information found", "Data not available", or "Details were limited"; the model must return `""` or `[]` instead.
     - **Human message**: provides the company name, combined texts for each section, and the candidate source URLs. It also explicitly instructs:
       - "Identify competitors based on the inferred industry. If the company is a tech company, list tech competitors. If it is a cybersecurity company, list cybersecurity competitors. Only use the content provided."
       - This ensures competitor extraction is driven by the **inferred industry from Tavily content**, not a hard-coded assumption (e.g., not always cybersecurity).

4. **Post-processing and schema enforcement**
   - The raw LLM output is parsed with `json.loads`; if parsing fails, a best-effort slice between the outermost `{`/`}` is attempted.
   - The code enforces defaults:
     - `company` defaults to the input company name.
     - `summary` → `""`, `products` → `[]`, `financials` → `""`, `competitors` → `[]`, `sources` → `[]` if missing.
   - `sources` is then overwritten with the Tavily-derived URL list `sources_urls`.
   - The resulting dict is returned as the `payload` in a `HandoffMessage` to the Supervisor with `status="completed"` and `next_agent=SUPERVISOR`.

**Important behavioral constraint:**
- Competitors are not hard-coded or forced to cybersecurity companies; they are derived solely from the Tavily answers, result contents, and provided URLs, with the LLM instructed to infer the company’s industry (e.g., tech vs cybersecurity) from that content.

### DocumentCreatorAgent (`agents/document_creator.py`)

**Responsibility:** convert structured research JSON into a markdown report and write it to disk.

Key elements:

- Uses a `ChatPromptTemplate` with a system message that enforces:
  - The model must only use information present in the input JSON and must not fabricate products, numbers, or boilerplate.
  - If a field is an empty string or empty list, that section may be omitted instead of filled with placeholders.
  - `sources` handling is strict:
    - If `research["sources"]` is a non-empty list, the model must add a `## Sources` section and render each URL as a bullet `- URL`.
    - If `sources` is empty or missing, the model must still include a `## Sources` section containing exactly the sentence `No sources were provided.` (this exact wording is important).
    - The phrase `Details were limited in the research results.` must never appear in the report.
- `run(...)`:
  - Validates that `next_agent == AgentName.DOCUMENT_CREATOR` and `task_name == "create_report"`.
  - Expects `payload` to contain `company_name` and a `research` dict (the Researcher’s JSON).
  - Dumps the research dict to pretty JSON and feeds it to the LLM formatting chain.
  - Calls `utils.file_writer.write_report_file(...)` with the generated markdown and returns a `HandoffMessage` with `status="completed"` and `file_path` set.

### Entry point (`main.py`)

- Wires up a shared `ChatOpenAI` model instance and the three agents.
- Loads environment variables (e.g., from `.env`) and prompts the user for a company name.
- Constructs the initial `HandoffMessage` for the Supervisor and runs the workflow, printing the final report path on success.

### Files and directories

- `agents/` – implementations of `SupervisorAgent`, `ResearcherAgent`, and `DocumentCreatorAgent`.
- `utils/` – shared utilities:
  - `message_schema.py` – the handoff message structure and agent enum.
  - `file_writer.py` – helper for writing report files into `outputs/`.
- `outputs/` – target directory for generated markdown reports.

This high-level overview should be enough for future WARP instances to quickly understand how to plug into the workflow, where to modify prompts or behavior, and which commands to run when developing in this repository.
