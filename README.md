# Opay Review Sentiment Classifier

Classifies Opay app reviews as **positive**, **negative**, or **neutral**. Compares a fine-tuned BERT model against a TF-IDF + Random Forest baseline, deployed as a **Streamlit app** and a **FastAPI service**, containerized with Docker.

## Why This Matters

Fintech apps like Opay get thousands of app store reviews. Reading them manually doesn't scale. This project automates sentiment classification so teams can spot problems, track satisfaction trends, and prioritize urgent complaints without manually reading every review.

## Dataset

~10,000 Opay reviews, labeled by star rating (1–2 stars → negative, 3 stars → neutral, 4–5 stars → positive). Positive reviews dominate the dataset.

## Approach

1. Clean text (lowercase, remove punctuation/emojis)
2. Baseline: TF-IDF + Random Forest
3. Compare against: fine-tuned BERT (last 1–2 layers unfrozen)
4. Handle class imbalance with class weights and SMOTE
5. Evaluate with precision, recall, F1, and confusion matrices
6. Deploy the better model via Streamlit and FastAPI

## Results

| Metric | BERT | Random Forest |
|---|---|---|
| Accuracy | 0.80 | **0.82** |
| Macro F1 | 0.52 | **0.55** |

Random Forest matched BERT while being cheaper and faster to run, so it's the deployed model.

## Getting Started

**Streamlit app:**
```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

**FastAPI service:**
```bash
pip install -r requirements.txt
uvicorn fast_app:app --reload
```
Docs at `http://127.0.0.1:8000/docs`

**Docker:**
```bash
docker build -t opay-sentiment-api .
docker run -p 8000:8000 opay-sentiment-api
```

## API Example

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"text": "This app is fast and reliable, love it!"}'
```

```json
{
  "predicted_sentiment": "positive",
  "confidence": 0.87,
  "message": "Thank you for the kind words! We're glad Opay is working well for you."
}
```

`POST /predict/batch` handles up to 100 reviews at once.

## Real-World Uses

- Flag negative reviews for faster customer support response
- Track sentiment trends across app releases
- Benchmark against competitor app reviews
- Feed a product team dashboard

## Limitations

- Neutral/mixed reviews are classified less accurately than clear positive/negative ones
- Trained on English text; may struggle with Nigerian Pidgin or code-mixed reviews
- Specific to Opay's review data — needs retraining for other domains

## Next Steps

- Treat mixed-sentiment reviews as multi-label instead of single-label
- Get human-annotated labels to validate star-derived ones
- Add automated tests (pytest)
- Monitor for sentiment drift in production
