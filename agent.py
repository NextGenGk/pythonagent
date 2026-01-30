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
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
MCP_SERVER_URL = os.getenv("MCP_SERVER_URL", "https://pymcp.vercel.app")

# Setup FastAPI
app = FastAPI(title="Doctor AI Agent Service", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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
        with httpx.Client() as client:
            resp = client.post(url, json=payload, timeout=20.0)
            resp.raise_for_status()
            data = resp.json()
            
            if "error" in data:
                return f"Error fetching data: {data['error']}"
            
            content = data.get("result", {}).get("content", [])
            if content and len(content) > 0:
                return content[0].get("text", "No data text found")
            return "No data returned (Empty result)"
            
    except Exception as e:
        return f"Failed to connect to MCP Server: {str(e)}"

# Setup LLM with Tools
# We use models/gemini-2.5-flash as it was verified in health check
llm = ChatGoogleGenerativeAI(model="models/gemini-2.5-flash", google_api_key=GOOGLE_API_KEY)
llm_with_tools = llm.bind_tools([fetch_patient_data_tool])

class GenerateRequest(BaseModel):
    pid: str
    prompt: str

@app.post("/generate")
async def generate_prescription(req: GenerateRequest):
    """
    Agent Logic:
    1. Verify Intent (ArmorIQ)
    2. Decision: Need data? (LLM)
    3. Action: Fetch data (Tool)
    4. Synthesis: Write prescription (LLM)
    """
    security_status = False
    
    # --- 1. ArmorIQ Secure Intent ---
    if armoriq:
        try:
            secure_prompt = f"Doctor Request: {req.prompt} for Patient {req.pid}."
            armoriq.capture_plan(llm="gpt-4", prompt=secure_prompt)
            security_status = True
        except Exception as e:
            print(f"ArmorIQ Failed: {e}")

    # --- 2. Manual Agent Loop (Requested Flow) ---
    try:
        messages = [
            HumanMessage(
                content=f"You are a doctor assistant. Instructions: {req.prompt}. You MUST fetch patient data for PID '{req.pid}' before generating the prescription."
            )
        ]
        
        # Step A: Agent Decides/Calls Tool
        response_1 = await llm_with_tools.ainvoke(messages)
        messages.append(response_1)
        
        final_content = response_1.content

        # Step B: Execute Tool if requested
        if response_1.tool_calls:
            for tool_call in response_1.tool_calls:
                if tool_call["name"] == "fetch_patient_data_tool":
                    # Call Tool
                    pid_arg = tool_call["args"].get("pid", req.pid)
                    tool_output = fetch_patient_data_tool.invoke({"pid": pid_arg})
                    
                    # Add Tool Output to conversation
                    messages.append(ToolMessage(content=tool_output, tool_call_id=tool_call["id"]))
            
            # Step C: Generate Final Answer
            response_2 = await llm_with_tools.ainvoke(messages)
            final_content = response_2.content
        
        return {
            "status": "success",
            "prescription_content": final_content,
            "security_verified": security_status
        }
    except Exception as e:
        print(f"Agent Loop Error: {e}")
        raise HTTPException(status_code=500, detail=f"Agent Error: {str(e)}")

@app.get("/health")
async def health():
    return {"status": "ok", "service": "doctor-agent", "model": "gemini-2.5-flash"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
