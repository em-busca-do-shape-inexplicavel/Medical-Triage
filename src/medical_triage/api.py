from contextlib import asynccontextmanager
from pathlib import Path

import joblib
from fastapi import FastAPI

from fastapi import Request
from pydantic import BaseModel, Field, field_validator

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "models" / "medical_triage_model.joblib"


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Model artifact not found: {MODEL_PATH}")

    artifact = joblib.load(MODEL_PATH)

    app.state.pipeline = artifact["pipeline"]
    app.state.label_mapping = artifact["label_mapping"]

    yield

    del app.state.pipeline
    del app.state.label_mapping

class PredictionRequest(BaseModel):
    text: str = Field(min_length=1)

    @field_validator("text")
    @classmethod
    def normalize_text(cls, value: str) -> str:
        normalized = " ".join(value.split())

        if not normalized:
            raise ValueError("Text must not be empty or whitespace only.")

        return normalized

class PredictionResponse(BaseModel):
    condition_label: int
    condition_name: str


app = FastAPI(
    title="Medical Text Classification API",
    description="Classifies English medical abstracts into five categories.",
    version="0.1.0",
    lifespan=lifespan,
)

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(
    payload: PredictionRequest,
    request: Request,
) -> PredictionResponse:
    pipeline = request.app.state.pipeline
    label_mapping = request.app.state.label_mapping

    prediction = pipeline.predict([payload.text])
    label = int(prediction[0])

    return PredictionResponse(
        condition_label=label,
        condition_name=label_mapping[label],
    )