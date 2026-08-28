from pydantic import BaseModel


class StepRequestBody(BaseModel):
    id: int
    action: str


class ResetRequestBody(BaseModel):
    id: int
    data_idx: int = 0


class CloseRequestBody(BaseModel):
    id: int


class FilesystemCheckpointCommitRequestBody(BaseModel):
    id: int
    session_index: int
    step_count: int
    size_bytes: int
    sha256: str


class WorkspaceInterventionRequestBody(BaseModel):
    id: int
    arm: str
    source_env_id: int | None = None


class WorkspaceExportRequestBody(BaseModel):
    id: int
