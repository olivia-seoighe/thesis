from typing import List, Optional
from pydantic import BaseModel


class SummarizeRequest(BaseModel):
    file_path: str
    content: str
    language: str = "csharp"
    url: str = ""
    last_modified: Optional[str] = None
    model: Optional[str] = None  # overrides OPENAI_MODEL env var


class SummarizeResponse(BaseModel):
    file_path: str
    url: str
    summary: str
    model_used: str
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None


class MigrationFile(BaseModel):
    file_path: str
    content: str


class MigrationAggregateRequest(BaseModel):
    repo: str
    files: List[MigrationFile]  # ordered oldest→newest by caller
    model: Optional[str] = None
