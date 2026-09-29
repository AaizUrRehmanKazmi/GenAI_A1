"""Proposed response contract; finalize when model tensor contracts are known."""
from pydantic import BaseModel, Field

class InferenceResult(BaseModel):
    image_base64: str
    inference_ms: float = Field(ge=0)
    probabilities: dict[str, float] | None = None
    routing_weights: dict[str, float] | None = None
    selected_expert: str | None = None
