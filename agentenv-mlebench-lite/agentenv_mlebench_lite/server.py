from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, StrictInt, StrictStr

from .environment import ActionSequenceError, EpisodeStep, MLEBenchLiteEpisodeManager


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class CreateRequest(_StrictModel):
    mode: Literal["native", "amg_memory"]


class ResetRequest(_StrictModel):
    id: StrictInt
    capability_token: StrictStr
    data_idx: StrictInt


class StepRequest(_StrictModel):
    id: StrictInt
    capability_token: StrictStr
    action_id: StrictStr
    action: StrictStr
    expected_action_count: StrictInt
    control: Literal["compaction"] | None = None


class CloseRequest(_StrictModel):
    id: StrictInt
    capability_token: StrictStr


def create_app(manager: MLEBenchLiteEpisodeManager) -> FastAPI:
    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        yield
        manager.close_all()

    app = FastAPI(
        title="MLE-bench Lite",
        docs_url=None,
        redoc_url=None,
        lifespan=lifespan,
    )

    @app.get("/metadata")
    def metadata() -> dict[str, Any]:
        return manager.metadata()

    @app.post("/create")
    def create(request: CreateRequest) -> dict[str, Any]:
        slot_id = manager.create(mode=request.mode)
        return {
            "id": slot_id,
            "capability_token": manager.capability_token(slot_id),
        }

    @app.post("/reset")
    def reset(request: ResetRequest) -> dict[str, Any]:
        try:
            return _public_step(
                manager.reset(
                    request.id,
                    request.data_idx,
                    capability_token=request.capability_token,
                )
            )
        except KeyError as exc:
            raise HTTPException(
                status_code=404, detail="environment slot unavailable"
            ) from exc
        except (IndexError, ValueError) as exc:
            raise HTTPException(status_code=422, detail="request rejected") from exc
        except Exception as exc:
            raise HTTPException(
                status_code=503, detail="environment unavailable"
            ) from exc

    @app.post("/step")
    def step(request: StepRequest) -> dict[str, Any]:
        try:
            return _public_step(
                manager.step(
                    request.id,
                    request.action,
                    action_id=request.action_id,
                    capability_token=request.capability_token,
                    control=request.control,
                    expected_action_count=request.expected_action_count,
                )
            )
        except KeyError as exc:
            raise HTTPException(
                status_code=404, detail="environment slot unavailable"
            ) from exc
        except ActionSequenceError as exc:
            raise HTTPException(
                status_code=409, detail="action sequence rejected"
            ) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail="episode unavailable") from exc
        except Exception as exc:
            raise HTTPException(
                status_code=503, detail="environment unavailable"
            ) from exc

    @app.post("/close")
    def close(request: CloseRequest) -> dict[str, bool]:
        try:
            manager.close(
                request.id,
                capability_token=request.capability_token,
            )
        except KeyError as exc:
            raise HTTPException(
                status_code=404, detail="environment slot unavailable"
            ) from exc
        except Exception as exc:
            raise HTTPException(
                status_code=503, detail="environment unavailable"
            ) from exc
        return {"closed": True}

    return app


def _public_step(step: EpisodeStep) -> dict[str, Any]:
    return {
        "observation": step.observation,
        "reward": step.reward,
        "done": step.done,
        "info": dict(step.info),
    }
