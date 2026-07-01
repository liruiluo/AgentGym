from pydantic import BaseModel


class StepRequestBody(BaseModel):
    id: int
    action: str


class ResetRequestBody(BaseModel):
    id: int
    data_idx: int = 0


class CloseRequestBody(BaseModel):
    id: int
