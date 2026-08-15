from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, ConfigDict, StrictInt, StrictStr

from .environment import OpenMLEFastEpisodeManager


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class CreateRequest(_StrictModel):
    pass


class ResetRequest(_StrictModel):
    id: StrictInt
    data_idx: StrictInt


class StepRequest(_StrictModel):
    id: StrictInt
    action: StrictStr


class SlotRequest(_StrictModel):
    id: StrictInt


_configured_manager: OpenMLEFastEpisodeManager | None = None


def configure(manager: OpenMLEFastEpisodeManager) -> None:
    global _configured_manager
    if _configured_manager is not None and _configured_manager is not manager:
        raise RuntimeError("OpenMLE-fast server manager is already configured")
    _configured_manager = manager


def _manager() -> OpenMLEFastEpisodeManager:
    global _configured_manager
    if _configured_manager is None:
        from .launch import build_manager_from_environment

        _configured_manager = build_manager_from_environment()
    return _configured_manager


def create_app(
    bound_manager: OpenMLEFastEpisodeManager | None = None,
) -> FastAPI:
    def selected() -> OpenMLEFastEpisodeManager:
        return bound_manager if bound_manager is not None else _manager()

    @asynccontextmanager
    async def lifespan(_application: FastAPI):
        manager = selected()
        manager.reconcile_orphans()
        try:
            yield
        finally:
            receipt = manager.close_all()
            if receipt["failed"]:
                raise RuntimeError("OpenMLE-fast shutdown cleanup failed")

    application = FastAPI(
        title="OpenMLE-fast",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
        lifespan=lifespan,
    )

    @application.get("/")
    def health() -> dict[str, str]:
        return {"status": "ok", "domain_id": "openmle_fast"}

    @application.get("/metadata")
    def metadata() -> dict[str, Any]:
        return selected().metadata()

    @application.post("/create")
    def create(_body: CreateRequest) -> dict[str, Any]:
        slot_id = selected().create()
        return {"id": slot_id, "observation": "OpenMLE-fast slot created.", "info": {}}

    @application.post("/reset")
    def reset(body: ResetRequest) -> dict[str, Any]:
        try:
            return selected().reset(body.id, body.data_idx).as_dict()
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise HTTPException(
                status_code=400, detail="invalid reset request"
            ) from exc

    @application.post("/step")
    def step(body: StepRequest) -> dict[str, Any]:
        try:
            return selected().step(body.id, body.action).as_dict()
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="unknown slot") from exc
        except RuntimeError as exc:
            raise HTTPException(
                status_code=409, detail="episode is not active"
            ) from exc

    @application.get("/observation")
    def observation(id: int = Query(...)) -> str:
        try:
            return selected().observation(id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="unknown slot") from exc
        except RuntimeError as exc:
            raise HTTPException(
                status_code=409, detail="episode is not active"
            ) from exc

    @application.post("/horizon")
    def horizon(body: SlotRequest) -> dict[str, Any]:
        try:
            return selected().finalize_horizon(body.id).as_dict()
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="unknown slot") from exc
        except RuntimeError as exc:
            raise HTTPException(
                status_code=409, detail="episode is not active"
            ) from exc

    @application.post("/close")
    def close(body: SlotRequest) -> dict[str, Any]:
        try:
            return selected().close(body.id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="unknown slot") from exc

    return application


app = create_app()
