from fastapi import FastAPI
from pydantic import BaseModel
from agent import run_agent

app = FastAPI()

class Req(BaseModel):
    url: str
    vibe: str = "lofi"

@app.post("/generate")
def generate(req: Req):
    return run_agent(req.url, req.vibe)
