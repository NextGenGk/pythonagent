import os
import httpx
import json
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import google.generativeai as genai

# Load Environment
load_dotenv()

# Configuration
ARMORIQ_API_KEY = os.getenv("ARMORIQ_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
MCP_SERVER_URL = os.getenv("MCP_SERVER_URL", "https://pymcp.vercel.app")

# Setup Gemini
if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)
    # Using the verified available model from list
    model = genai.GenerativeModel('models/gemini-2.5-flash')
else:
    model = None
    print("WARNING: GOOGLE_API_KEY not set")

# Setup ArmorIQ
armoriq = None
try:
    from armoriq_sdk import ArmorIQClient
    if ARMORIQ_API_KEY:
        armoriq = ArmorIQClient(api_key=ARMORIQ_API_KEY)
except ImportError:
    print("ArmorIQ SDK not installed")

# Setup FastAPI
app = FastAPI(title="Doctor AI Agent Service", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Relaxed for debug, update later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class GenerateRequest(BaseModel):
    pid: str
    prompt: str

async def call_mcp_tool(pid: str):
    """Call the deployed MCP tool"""
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
    async with httpx.AsyncClient() as client:
        resp = await client.post(url, json=payload, timeout=30.0)
        data = resp.json()
        content = data.get("result", {}).get("content", [])
        if content:
            return content[0].get("text", "No data")
        return "No patient data found."

@app.post("/generate")
async def generate_prescription(req: GenerateRequest):
    if not model:
        raise HTTPException(status_code=500, detail="Gemini model not configured")

    security_verified = False
    # 1. ArmorIQ Secure Intent
    if armoriq:
        try:
            armoriq.capture_plan(llm="gemini-2.5-flash", prompt=f"Generate prescription for {req.pid}: {req.prompt}")
            security_verified = True
        except:
            pass

    # 2. Logic: Fetch Data -> Generate Text
    # We do a direct 2-step process for reliability
    try:
        # Step A: Fetch Data
        patient_info = await call_mcp_tool(req.pid)
        
        # Step B: Generate Prescription
        system_prompt = f"""
        Act as a professional doctor assistant. Generate a professional medical prescription.
        
        PATIENT DATA:
        {patient_info}
        
        DOCTOR'S INSTRUCTIONS:
        {req.prompt}
        
        Format the output clearly with:
        - Diagnosis
        - Medications (Name, Dosage, Frequency)
        - Advice
        """
        
        response = model.generate_content(system_prompt)
        
        return {
            "status": "success",
            "prescription_content": response.text,
            "security_verified": security_verified
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent Error: {str(e)}")

@app.get("/health")
async def health():
    try:
        models = [m.name for m in genai.list_models()]
    except Exception as e:
        models = f"Error listing models: {str(e)}"
    return {
        "status": "ok", 
        "service": "doctor-agent", 
        "mcp_url": MCP_SERVER_URL,
        "available_models": models
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
