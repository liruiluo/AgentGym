from fastapi import FastAPI, Header, HTTPException

from .model import (
    CloseRequestBody,
    FilesystemCheckpointCommitRequestBody,
    ResetRequestBody,
    StepRequestBody,
    WorkspaceExportRequestBody,
    WorkspaceInterventionRequestBody,
)
from .runtime.server_factory import build_server
from .service_identity import decorate_service_metadata


server = build_server()

app = FastAPI()


@app.get("/")
def hello() -> str:
    return "This is environment AgentMemoryGym."


@app.get("/metadata")
def metadata():
    return decorate_service_metadata(server.metadata())


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


@app.post("/filesystem-checkpoint-commit")
def filesystem_checkpoint_commit(body: FilesystemCheckpointCommitRequestBody):
    control = getattr(server, "filesystem_checkpoint_commit", None)
    if control is None:
        raise HTTPException(
            status_code=404,
            detail="filesystem checkpoint commit is unavailable on this surface",
        )
    try:
        return control(
            body.id,
            session_index=body.session_index,
            step_count=body.step_count,
            size_bytes=body.size_bytes,
            sha256=body.sha256,
        )
    except (KeyError, RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/workspace-intervention")
def workspace_intervention(
    body: WorkspaceInterventionRequestBody,
    token: str = Header(alias="X-AgentMemory-Intervention-Token"),
):
    control = getattr(server, "workspace_intervention", None)
    if control is None:
        raise HTTPException(
            status_code=404,
            detail="workspace intervention control is unavailable on this surface",
        )
    try:
        return control(
            body.id,
            arm=body.arm,
            source_env_id=body.source_env_id,
            token=token,
        )
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except (KeyError, RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/workspace-export")
def workspace_export(
    body: WorkspaceExportRequestBody,
    token: str = Header(alias="X-AgentMemory-Intervention-Token"),
):
    control = getattr(server, "workspace_export", None)
    if control is None:
        raise HTTPException(
            status_code=404,
            detail="workspace export control is unavailable on this surface",
        )
    try:
        return control(body.id, token=token)
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except (KeyError, RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
