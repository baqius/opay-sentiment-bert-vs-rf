import logging
from contextlib import asynccontextmanager
from typing import List

import joblib
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

from preprocessing import preprocessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("opay-sentiment-api")

MODEL_PATH = "rf_sentiment_model.pkl"
VECTORIZER_PATH = "tfidf_vectorizer.pkl"
ENCODER_PATH = "label_encoder.pkl"

# Container for loaded ML artifacts, populated at startup
ml_artifacts = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model artifacts once at startup and release them on shutdown."""
    logger.info("Loading model artifacts...")
    try:
        ml_artifacts["model"] = joblib.load(MODEL_PATH)
        ml_artifacts["vectorizer"] = joblib.load(VECTORIZER_PATH)
        ml_artifacts["encoder"] = joblib.load(ENCODER_PATH)
        logger.info("Model artifacts loaded successfully.")
    except FileNotFoundError as e:
        logger.error(f"Failed to load model artifacts: {e}")
        raise
    yield
    logger.info("Shutting down. Clearing model artifacts from memory.")
    ml_artifacts.clear()


app = FastAPI(
    title="Opay Review Sentiment Classifier API",
    description=(
        "Predicts whether an Opay app review is positive, negative, or "
        "neutral using a TF-IDF + Random Forest model."
    ),
    version="1.0.0",
    lifespan=lifespan,
)



class ReviewRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="The review text to classify.",
        examples=["This app is fast and reliable, love it!"],
    )


class BatchReviewRequest(BaseModel):
    texts: List[str] = Field(
        ...,
        min_length=1,
        max_length=100,
        description="A list of review texts to classify (max 100 per request).",
    )


class SentimentConfidence(BaseModel):
    label: str
    probability: float


class PredictionResponse(BaseModel):
    text: str
    predicted_sentiment: str
    confidence: float
    is_low_confidence: bool
    breakdown: List[SentimentConfidence]


class BatchPredictionResponse(BaseModel):
    results: List[PredictionResponse]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool




LOW_CONFIDENCE_THRESHOLD = 0.5


def predict_sentiment(text: str) -> PredictionResponse:
    """Run the full preprocessing + inference pipeline on a single review."""
    if not text or not text.strip():
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Review text must not be empty.",
        )

    model = ml_artifacts["model"]
    vectorizer = ml_artifacts["vectorizer"]
    encoder = ml_artifacts["encoder"]

    cleaned_text = preprocessor(text)
    text_tfidf = vectorizer.transform([cleaned_text])

    prediction = model.predict(text_tfidf)
    predicted_label = encoder.inverse_transform(prediction)[0]

    probabilities = model.predict_proba(text_tfidf)[0]
    breakdown = [
        SentimentConfidence(label=label, probability=round(float(prob), 4))
        for label, prob in zip(encoder.classes_, probabilities)
    ]

    top_confidence = float(max(probabilities))

    return PredictionResponse(
        text=text,
        predicted_sentiment=predicted_label,
        confidence=round(top_confidence, 4),
        is_low_confidence=top_confidence < LOW_CONFIDENCE_THRESHOLD,
        breakdown=breakdown,
    )



@app.get("/", tags=["General"])
def root():
    return {
        "message": "Opay Review Sentiment Classifier API",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
def health_check():
    """Simple readiness check confirming the model is loaded and ready to serve."""
    return HealthResponse(
        status="ok",
        model_loaded="model" in ml_artifacts,
    )


@app.post(
    "/predict",
    response_model=PredictionResponse,
    tags=["Prediction"],
    summary="Predict sentiment for a single review",
)
def predict(request: ReviewRequest):
    """Classify a single Opay review as positive, negative, or neutral."""
    try:
        return predict_sentiment(request.text)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Prediction failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {e}",
        )


@app.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
    tags=["Prediction"],
    summary="Predict sentiment for multiple reviews at once",
)
def predict_batch(request: BatchReviewRequest):
    """Classify a batch of Opay reviews (up to 100 per request)."""
    try:
        results = [predict_sentiment(text) for text in request.texts]
        return BatchPredictionResponse(results=results)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Batch prediction failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction failed: {e}",
        )