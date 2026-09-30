from typing import Literal
from pydantic import BaseModel, ConfigDict, Field


class MoleculeInput(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    smiles: str = Field(min_length=1, max_length=2000)
    name: str = Field(default="", max_length=120)


class PreviewRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str = Field(min_length=1, max_length=220000)


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    mode: Literal["single", "batch", "mixture"]
    molecules: list[MoleculeInput] = Field(min_length=1, max_length=100)
    acknowledge_storage: Literal[True]


class InputProblem(Exception):
    def __init__(self, code: str, zh: str, en: str, issues=None, status=422):
        self.code, self.zh, self.en = code, zh, en
        self.issues = issues or []
        self.status = status
        super().__init__(en)

    def payload(self):
        return {"code": self.code, "message": {"zh": self.zh, "en": self.en}, "issues": self.issues}
