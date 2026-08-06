from __future__ import annotations

import logging
import threading
from typing import Any

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel

from .environment import SwesmithEpisodeManager
from .privacy import private_detail_authorized


app = FastAPI(debug=False, title="AgentMemoryGym SWE-smith")
_manager: SwesmithEpisodeManager | None = None
_manager_lock = threading.Lock()


class ResetRequest(BaseModel):
    id: int
    data_idx: int


class StepRequest(BaseModel):
    id: int
    action: str


class CloseRequest(BaseModel):
    id: int


def configure(manager: SwesmithEpisodeManager) -> None:
    global _manager
    with _manager_lock:
        if _manager is not None and _manager is not manager:
            raise RuntimeError("SWE-smith server manager is already configured")
        _manager = manager


def manager() -> SwesmithEpisodeManager:
    global _manager
    with _manager_lock:
        if _manager is None:
            from .launch import build_manager_from_environment

            _manager = build_manager_from_environment()
        return _manager


@app.get("/")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/metadata")
def metadata() -> dict[str, Any]:
    return manager().metadata()


@app.post("/create")
def create() -> dict[str, Any]:
    slot_id = manager().create()
    return {
        "id": slot_id,
        "observation": "SWE-smith environment created; reset with an explicit data_idx.",
        "reward": 0.0,
        "done": False,
        "info": {"schema": "agentmemory_swesmith_native_episode_v1"},
    }


@app.post("/reset")
def reset(body: ResetRequest) -> dict[str, Any]:
    try:
        return manager().reset(body.id, body.data_idx).as_dict()
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logging.exception("SWE-smith reset failed")
        raise HTTPException(
            status_code=500,
            detail="SWE-smith reset failed closed; inspect server logs",
        ) from exc


@app.post("/step")
def step(body: StepRequest) -> dict[str, Any]:
    try:
        return manager().step(body.id, body.action).as_dict()
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except Exception as exc:
        logging.exception("SWE-smith step failed")
        raise HTTPException(
            status_code=500,
            detail="SWE-smith step failed closed; inspect server logs",
        ) from exc


@app.get("/observation")
def observation(id: int) -> str:
    try:
        return manager().observation(id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.get("/detail")
def detail(
    id: int,
    x_swesmith_detail_token: str | None = Header(default=None),
) -> dict[str, Any]:
    if not private_detail_authorized(x_swesmith_detail_token):
        # Hide the existence of the private route from ordinary clients.
        raise HTTPException(status_code=404, detail="Not found")
    try:
        return manager().detail(id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.post("/close")
def close(body: CloseRequest) -> dict[str, Any]:
    try:
        return manager().close(body.id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        logging.exception("SWE-smith close failed")
        raise HTTPException(
            status_code=500,
            detail="SWE-smith close failed closed; inspect server logs",
        ) from exc
