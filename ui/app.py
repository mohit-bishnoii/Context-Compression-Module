import os
import sys
import json
from datetime import datetime

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from travel_agent.agent import CCMAgent

app = FastAPI()
templates = Jinja2Templates(directory=os.path.join(os.path.dirname(__file__), "templates"))

agent = CCMAgent()
agent.reset()
logs = []
conversation = []

@app.get("/", response_class=HTMLResponse)
async def main(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "conversation": conversation,
            "logs": logs,
            "token_amount": agent.token_counts[-1] if agent.token_counts else 0,
            "memory_state": agent.ccm.get_memory_state(),
        }
    )

@app.post("/chat")
async def chat(request: Request):
    global conversation, logs
    
    data = await request.form()
    user_message = data.get("message", "").strip()
    
    if not user_message:
        return {"status": "error", "message": "Empty message"}
    
    timestamp = datetime.now().strftime("%H:%M:%S")
    logs.append(f"[{timestamp}] User: {user_message}")
    
    result = agent.chat(user_message)
    
    conversation.append({
        "role": "user",
        "content": user_message,
        "timestamp": timestamp,
    })
    conversation.append({
        "role": "assistant",
        "content": result.get("response", ""),
        "timestamp": timestamp,
    })
    
    token_count = result.get("tokens_in_context", 0)
    logs.append(f"[{timestamp}] Tokens: {token_count} | Turn: {result.get('turn_number', 0)}")
    logs.append(f"[{timestamp}] Agent: {result.get('response', '')[:100]}...")
    
    if len(logs) > 100:
        logs = logs[-100:]
    
    return {
        "status": "ok",
        "response": result.get("response", ""),
        "tokens": token_count,
        "turn": result.get("turn_number", 0),
        "memory_state": agent.ccm.get_memory_state(),
    }

@app.post("/reset")
async def reset():
    global conversation, logs
    agent.reset()
    conversation = []
    logs = []
    return {"status": "ok", "message": "Reset complete"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)