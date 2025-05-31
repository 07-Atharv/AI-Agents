from pydantic import BaseModel
from typing import List
from fastapi import FastAPI
from ai_agents import response_from_ai_agent
import uvicorn

#Manual class will inherit the base model
class RequestState(BaseModel):
    model_name: str
    model_provider: str
    system_prompt: str
    prompt: str
    allow_search:bool
    
ALLOWED_MODEL_NAMES=["llama-3.3-70b-versatile","llama3-70b-8192","mistral-8x7b-32768","gpt-4o-mini"]
    
app=FastAPI(title="Langgraph test app")
@app.post("/chat")
def chat_endpoint(request:RequestState):
    if request.model_name not in ALLOWED_MODEL_NAMES:
        return {"Error":"Test test"}
    
    llm_id=request.model_name
    query=request.prompt
    allow_search=request.allow_search
    prompt=request.system_prompt
    provider=request.model_provider
    
    response=response_from_ai_agent(llm_id,query,allow_search,prompt,provider)
    return response
    
    
if __name__=="__main__":
    uvicorn.run(app,host="127.0.0.1",port=9000)