# Multi-Agent Handoff Research System

This project is a minimal, end-to-end example of a **multi-agent system with strict handoff** built on **LangChain**.

Three agents collaborate to complete a company research task:

1. **Supervisor Agent** – Understands the user's request, coordinates the workflow, and orchestrates handoffs.
2. **Researcher Agent** – Gathers and structures company information.
3. **Document Creator Agent** – Turns structured research into a formatted report file.

Workflow (fixed order):

```text
User → Supervisor → Researcher → Supervisor → Document Creator → Supervisor → User
```

Each agent has a **single responsibility**, and only the Supervisor coordinates the others.

---

## Project Structure

```text
.
├── main.py
├── README.md
├── requirements.txt
├── .gitignore
├── agents/
│   ├── supervisor.py
│   ├── researcher.py
│   └── document_creator.py
├── utils/
│   ├── message_schema.py
│   └── file_writer.py
└── outputs/
```

- `agents/` – Implementation of the Supervisor, Researcher, and Document Creator agents.
- `utils/message_schema.py` – Shared structured message type for handoffs.
- `utils/file_writer.py` – Helper for writing report files into the `outputs/` folder.
- `outputs/` – Generated reports are saved here.

---

## Setup

It is recommended to use a Python virtual environment.

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Environment Variables

This project expects an LLM compatible with LangChain. By default it uses **OpenAI** via `langchain-openai`.
It also uses the **Tavily Search API** via `langchain-community` for real web research in the Researcher agent.

Create a `.env` file in the project root:

```bash
OPENAI_API_KEY="your-openai-api-key"
OPENAI_MODEL="gpt-4o-mini"      # optional, defaults to gpt-4o-mini
OPENAI_TEMPERATURE="0.2"        # optional
TAVILY_API_KEY="your-tavily-api-key"  # required for web research
```

> You can switch to other LangChain-supported models (e.g., Groq) by adjusting the model class in `main.py` and the corresponding environment variables.

---

## Running the Application

From the project root:

```bash
source venv/bin/activate
python main.py
```

You will be prompted for a company name, for example:

```text
Enter a company name to research: Check Point
```

The agents will coordinate via the Supervisor to:

1. Research the company.
2. Produce structured JSON research.
3. Generate a markdown report.
4. Save the file in `outputs/`.

On success, you'll see output similar to:

```text
Research report successfully created!
Report file: /absolute/path/to/project/outputs/check-point-20250101-120000.md
```

---

## Implementation Notes

### Message Schema

All agents exchange a common `HandoffMessage` structure defined in `utils/message_schema.py`:

- `task_name` – logical name of the current task.
- `payload` – task-specific data (company name or research data).
- `next_agent` – one of `supervisor`, `researcher`, `document_creator`, or `None`.
- `status` – `pending`, `in_progress`, `completed`, or `failed`.
- `file_path` – path to the generated report (set by Document Creator).
- `error` – error details, if any.

The **Supervisor** uses this schema to coordinate transitions and enforce the strict handoff order.

### LangChain-Based Agents

Each agent uses **LangChain** in its own way:

- **Supervisor** – Uses a small LLM chain to interpret the user's high-level request, but delegates all work.
- **Researcher** – Uses `TavilySearchResults` (via `langchain-community`) to perform multiple focused web searches (overview, products, financials, competitors, news) and feeds those results into an LLM chain that returns strict JSON with keys: `company`, `summary`, `products`, `financials`, `competitors`, `sources`.
- **Document Creator** – Uses an LLM chain to turn the JSON research into a clean markdown report and writes it using `file_writer`, without fabricating any information beyond what is present in the JSON.

All three share the same `ChatOpenAI` model instance for efficiency.

---

## Testing Guidance

The codebase is structured to make it easy to add tests (e.g., with `pytest`):

1. **Researcher output quality**
   - Instantiate `ResearcherAgent` with a **mock or test double model** that returns deterministic JSON.
   - Call `run(...)` with a `HandoffMessage` containing a test company name.
   - Assert that:
     - `status == "completed"`
     - The `payload` JSON has the keys: `company_summary`, `products`, `financials`, `competitors`, `sources`.

2. **Document file creation**
   - Instantiate `DocumentCreatorAgent` with a mock model returning a fixed markdown string.
   - Call `run(...)` with a message whose payload includes a fake research dict.
   - Assert that:
     - `status == "completed"`
     - `file_path` is non-empty and points to an existing file.

3. **Full workflow execution**
   - Use the `run_supervisor` helper in `main.py` with stubbed agents or a stubbed model.
   - Assert the supervisor returns a `HandoffMessage` with `status == "completed"` and a non-empty `file_path`.

4. **Incorrect or missing inputs**
   - Pass messages missing `company_name` or `research` to each agent.
   - Assert `status == "failed"` and that `error` is populated.

---

## Requirements

Key dependencies (see `requirements.txt`):

- `langchain`
- `langchain-community`
- `langchain-openai` (or similar LLM provider)
- `python-docx`
- `python-dotenv`
- `requests`

Install them with:

```bash
pip install -r requirements.txt
```

---

## Notes

- The included web search function in `ResearcherAgent` is intentionally minimal and should be swapped for production-grade tooling.
- The design focuses on **clear separation of responsibilities** and a **strict, inspectable handoff chain** between agents.
