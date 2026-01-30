# Doctor Agent Service Implementation Plan

## Overview

We are creating a dedicated "Agent Service" (`doctor-agent`) that acts as the intelligent orchestration layer. It separates the "Decision Making" (Agent) from the "Data Access" (MCP Tool).

## Architecture

| Component        | Port | Role                             | Technology                  |
| ---------------- | ---- | -------------------------------- | --------------------------- |
| **Doctor View**  | 3001 | Frontend UI                      | Next.js                     |
| **Doctor Agent** | 8001 | **The Brain** (Logic & Security) | FastAPI, LangChain, ArmorIQ |
| **MCP Server**   | 8000 | **The Hands** (Data Access)      | FastAPI, Supabase           |

## Folder Structure `doctor-agent/`

```
doctor-agent/
├── agent.py              # Main application logic
├── requirements.txt      # Dependencies (LangChain, ArmorIQ)
├── .env                  # API Keys (ArmorIQ, Gemini)
├── implementation.md     # This file
└── start_agent.bat       # Startup script
```

## Implementation Steps

### 1. Environment Setup

- Create folder `doctor-agent` (Done).
- Create `requirements.txt` with: `fastapi`, `uvicorn`, `langchain`, `langchain-google-genai`, `armoriq-sdk`.
- Setup `.env` with keys copied from MCP server.

### 2. The Agent Logic (`agent.py`)

The agent will expose a POST endpoint `/generate`.

**Workflow:**

1.  **Receive Request**: `pid`, `prompt`.
2.  **ArmorIQ Security Check**:
    - Call `client.capture_plan(prompt)` to verify intent.
    - Result: A verified plan confirming we should "fetch data" and "generate prescription".
3.  **LangChain Execution**:
    - Initialize Gemini Chat Model.
    - Define `fetch_patient_data` as a LangChain Tool (which calls `http://localhost:8000/mcp`).
    - Run the Agent with the user prompt.
    - Agent automatically decides to call the tool.
4.  **Return**: The final generated prescription.

### 3. Integration

- Update `doctor_view` frontend to call `http://localhost:8001/generate` (Agent) instead of `8000` (Data Tool).

## Why this is better?

- **Decoupled**: Your data tool stays simple (just fetches data). Your agent handles the complex logic.
- **Scalable**: You can add more tools to the agent later without touching the database logic.
- **Secure**: ArmorIQ sits exactly where it belongs—verifying the Agent's intent before it touches the tools.
