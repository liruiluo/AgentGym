from __future__ import annotations

import logging
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, StrictInt, StrictStr

from .wrapper import GaiaTextEpisodeManager

_LOGGER = logging.getLogger(__name__)


class ResetRequest(BaseModel):
    id: StrictInt
    data_idx: StrictInt


class StepRequest(BaseModel):
    id: StrictInt
    action: StrictStr


class EnvironmentRequest(BaseModel):
    id: StrictInt


def create_app(manager: GaiaTextEpisodeManager) -> FastAPI:
    app = FastAPI(
        debug=False,
        title="AgentMemoryGym GAIA-Text",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )

    @app.get("/")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/metadata")
    def metadata() -> dict[str, Any]:
        return manager.metadata()

    @app.post("/create")
    def create() -> dict[str, Any]:
        return manager.create()

    @app.post("/reset")
    def reset(body: ResetRequest) -> dict[str, Any]:
        return _call(manager.reset, body.id, body.data_idx)

    @app.post("/step")
    def step(body: StepRequest) -> dict[str, Any]:
        return _call(manager.step, body.id, body.action)

    @app.post("/horizon")
    def horizon(body: EnvironmentRequest) -> dict[str, Any]:
        return _call(manager.finalize_horizon, body.id)

    @app.post("/close")
    def close(body: EnvironmentRequest) -> bool:
        return _call(manager.close, body.id)

    return app


def _call(function, *args):
    try:
        return function(*args)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (IndexError, TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except Exception as exc:
        _LOGGER.exception("GAIA-Text environment request failed closed")
        raise HTTPException(
            status_code=500,
            detail="GAIA-Text request failed closed; inspect private server logs",
        ) from exc
