# api/schemas.py
from pydantic import BaseModel, Field
from typing import List, Optional

class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1)
    topk: Optional[int] = Field(None, ge=1, le=50, description="Override top-k if provided")
    threshold: Optional[float] = Field(None, ge=-1e9, le=1e9, description="Override threshold if provided")

class PredictResponse(BaseModel):
    tags: List[str]