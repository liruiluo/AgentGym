from fastapi import FastAPI

from .env_wrapper import server
from .model import CloseRequestBody, ResetRequestBody, StepRequestBody

app = FastAPI()


@app.get("/")
def hello() -> str:
    return "This is environment AgentMemoryGym."


@app.post("/create")
def create():
    return server.create()


@app.post("/step")
def step(body: StepRequestBody):
    return server.step(body.id, body.action)


@app.post("/reset")
def reset(body: ResetRequestBody):
    return server.reset(body.id, body.data_idx)


@app.get("/observation")
def observation(id: int):
    return server.observation(id)


@app.get("/detail")
def detail(id: int):
    return server.detail(id)


@app.post("/close")
def close(body: CloseRequestBody):
    return server.close(body.id)
