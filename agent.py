import os
import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# LangChain Imports - Minimal & Robust
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

# ArmorIQ Import
try:
    from armoriq_sdk import ArmorIQClient
except ImportError:
    ArmorIQClient = None
    print("WARNING: armoriq-sdk not found. Security features disabled.")

# Load Environment
load_dotenv()

# Configuration
ARMORIQ_API_KEY = os.getenv("ARMORIQ_API_KEY")
USER_ID = os.getenv("USER_ID", "doctor_admin")
AGENT_ID = os.getenv("AGENT_ID", "agent_gen")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
MCP_SERVER_URL = os.getenv("MCP_SERVER_URL", "http://localhost:8000")

# Setup FastAPI
app = FastAPI(title="Doctor AI Agent Service", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup ArmorIQ
armoriq = None
if ArmorIQClient and ARMORIQ_API_KEY:
    try:
        armoriq = ArmorIQClient(api_key=ARMORIQ_API_KEY)
    except Exception as e:
        print(f"Failed to init ArmorIQ: {e}")

# Tool Definition: Call MCP Server
@tool
def fetch_patient_data_tool(pid: str) -> str:
    """
    Fetches comprehensive patient medical data, including history, allergies, and current medications.
    Use this tool BEFORE generating any medical advice or prescription.
    """
    url = f"{MCP_SERVER_URL}/mcp"
    payload = {
        "jsonrpc": "2.0",
        "method": "tools/call",
        "params": {
            "name": "fetch_patient_data",
            "arguments": {"pid": pid}
        },
        "id": 1
    }
    
    try:
        # We use sync httpx for simplicity here
        with httpx.Client() as client:
            resp = client.post(url, json=payload, timeout=10.0)
            resp.raise_for_status()
            data = resp.json()
            
            if "error" in data:
                return f"Error fetching data: {data['error']}"
            
            # The result from MCP structure
            content = data.get("result", {}).get("content", [])
            if content and len(content) > 0:
                return content[0].get("text", "No data text found")
            return "No data returned (Empty result)"
            
    except Exception as e:
        return f"Failed to connect to MCP Server: {str(e)}"

# Setup LLM with Tools
# We use simple bind_tools which works on most recent langchain versions
llm = ChatGoogleGenerativeAI(model="models/gemini-1.5-flash", google_api_key=GOOGLE_API_KEY)
llm_with_tools = llm.bind_tools([fetch_patient_data_tool])

class GenerateRequest(BaseModel):
    pid: str
    prompt: str

@app.post("/generate")
async def generate_prescription(req: GenerateRequest):
    """
    Agent Endpoint with Manual Logic
    """
    security_status = False
    
    # --- 1. ArmorIQ Security Check disabled for debug ---
    security_status = False
    # if armoriq: ...

    # --- 2. Manual Agent Loop (Robust) ---
    try:
        messages = [
            HumanMessage(
                content=f"You are a doctor assistant. Fetch patient data for PID '{req.pid}' then generate a prescription based on: {req.prompt}"
            )
        ]
        
        # Call 1: LLM decides to use tool or output text
        print(f"Invoking LLM for First Step...")
        response_1 = await llm_with_tools.ainvoke(messages)
        messages.append(response_1)
        
        final_content = response_1.content

        # Check for tool calls
        if response_1.tool_calls:
            print(f"Tool Call Detected: {response_1.tool_calls}")
            for tool_call in response_1.tool_calls:
                # Execute Tool
                if tool_call["name"] == "fetch_patient_data_tool":
                    # Extract arg (LangChain returns dict in args)
                    pid_arg = tool_call["args"].get("pid")
                    
                    # Manual Invoke of the python function
                    tool_output = fetch_patient_data_tool.invoke({"pid": pid_arg})
                    
                    print(f"Tool Output: {str(tool_output)[:100]}...")
                    
                    # Append result to history
                    messages.append(ToolMessage(content=tool_output, tool_call_id=tool_call["id"]))
            
            # Call 2: LLM generates final answer given the tool output
            print("Invoking LLM for Final Answer...")
            response_2 = await llm_with_tools.ainvoke(messages)
            final_content = response_2.content
        
        return {
            "status": "success",
            "prescription_content": final_content,
            "security_verified": security_status
        }
    except Exception as e:
        print(f"Agent Loop Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    return {"status": "ok", "service": "doctor-agent", "model": "gemini-1.5-flash"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
