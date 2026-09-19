# Opay Review Sentiment Classifier — BERT vs. Random Forest

**Live Demo:** [https://opay-sentiment-bert-vs-rf.streamlit.app/](https://opay-sentiment-bert-vs-rf.streamlit.app/)

A sentiment analysis system that classifies Opay app reviews as **positive**, **negative**, or **neutral**, comparing a fine-tuned BERT model against a TF-IDF + Random Forest baseline. The better-performing model is deployed as both an interactive **Streamlit web app** and a production-ready **FastAPI service**, containerized with Docker.


## Overview

Opay is a mobile fintech platform, and its app store reviews contain a large volume of unstructured customer feedback. This project builds a system to automatically classify that feedback by sentiment, enabling faster, data-driven insight into customer satisfaction at scale — without manually reading thousands of reviews.

Two modeling approaches were built and rigorously compared:

1. **BERT** (partially fine-tuned, last encoder layers unfrozen)  a deep transformer model
2. **Random Forest** on TF-IDF features  a lightweight, classical ML baseline

The project also demonstrates a full ML lifecycle: data cleaning, exploratory analysis, model training and evaluation, class-imbalance handling, error analysis, and deployment via both a web UI and a REST API.

## Business Problem

Fintech companies like Opay handle millions of transactions and receive continuous user feedback through app store reviews. Manually monitoring this feedback doesn't scale, yet it contains valuable signals:

- Early warnings about bugs, outages, or security concerns
- Feature requests and usability friction points
- Overall brand sentiment trends over time

An automated sentiment classifier allows a product or customer experience team to triage this feedback in real time, prioritize urgent negative reviews, and track sentiment trends without manual review.

## Dataset

- ~10,000 Opay app reviews, each with review text and a sentiment label derived from the review's star rating (1–2 stars → negative, 3 stars → neutral, 4–5 stars → positive)
- Significant class imbalance: positive reviews dominate the dataset, while neutral reviews are a small minority

## Approach

1. **Text cleaning** — lowercasing, punctuation and emoji removal, tokenization
2. **Exploratory analysis** — class distribution, review length distribution (word and token counts), informing preprocessing decisions such as maximum sequence length
3. **Baseline model** — TF-IDF vectorization + Random Forest classifier, with stopword removal and lemmatization
4. **Transformer model** — BERT with a custom classification head; experimented with a fully frozen BERT (feature extractor only) versus partial fine-tuning (unfreezing the last 1–2 encoder layers plus the pooler) using discriminative learning rates
5. **Class imbalance handling** — class-weighted loss functions and oversampling (SMOTE, Random Oversampling) were tested against the persistent minority-class problem
6. **Evaluation** — precision, recall, F1-score, and confusion matrices, compared across both models
7. **Deployment** — the selected model was packaged behind a Streamlit UI and a FastAPI REST service, containerized with Docker for portability

## Results

| Metric | BERT | Random Forest |
|---|---|---|
| Accuracy | 0.80 | **0.82** |
| Macro F1 | 0.52 | **0.55** |
| Weighted F1 | 0.79 | 0.80 |

**Random Forest matched or slightly outperformed BERT** on this dataset — a deliberately reported, evidence-based outcome rather than an assumption that a larger model is always better. Given Random Forest's dramatically lower computational cost, faster inference, and simpler deployment footprint (no GPU, no deep learning framework), it was selected as the production model.

## Key Finding: The Class Imbalance Wasn't the Real Problem

Both models struggled significantly with the neutral class, and applying class weighting and SMOTE oversampling to the Random Forest model produced negligible improvement (neutral-class F1 remained ~0.08 regardless of technique).

Manual inspection of misclassified neutral reviews revealed the actual cause: **star-rating-derived neutral labels don't correspond to linguistically neutral text.** Reviews rated 3 stars typically contain clearly mixed sentiment (e.g., *"Good app, but the ATM card issuance is a nightmare"*) rather than flat, indifferent language. Since two structurally different models (a deep transformer and a tree ensemble) converged on the same failure mode despite different balancing techniques, this points to a **labeling scheme limitation**, not a model capacity or data volume problem.

This is a practically important, generalizable finding: **star ratings are a proxy for sentiment, not a ground-truth label for text sentiment**, and any real-world deployment using star-derived labels should account for this gap — for example, by treating "neutral" reviews as a distinct "mixed sentiment" category rather than a point on the positive–negative spectrum.

## Tech Stack

- **Modeling:** Python, scikit-learn, PyTorch, Hugging Face Transformers (BERT)
- **NLP preprocessing:** NLTK (stopwords, lemmatization)
- **Deployment:** Streamlit (interactive demo), FastAPI (REST API), Docker
- **Serialization:** joblib

## Getting Started

### Run the Streamlit app locally

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

### Run the FastAPI service locally

```bash
pip install -r requirements.txt
uvicorn fast_app:app --reload
```
Visit `http://127.0.0.1:8000/docs` for interactive API documentation.

### Run the API with Docker

```bash
docker build -t opay-sentiment-api .
docker run -p 8000:8000 opay-sentiment-api
```

## API Usage

**Single prediction:**
```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"text": "This app is fast and reliable, love it!"}'
```

**Response:**
```json
{
  "text": "This app is fast and reliable, love it!",
  "predicted_sentiment": "positive",
  "confidence": 0.87,
  "is_low_confidence": false,
  "breakdown": [
    {"label": "negative", "probability": 0.05},
    {"label": "neutral", "probability": 0.08},
    {"label": "positive", "probability": 0.87}
  ],
  "message": "Thank you for the kind words! We're glad Opay is working well for you."
}
```

**Batch prediction:** `POST /predict/batch` accepts up to 100 reviews in a single request.

## Real-World Applications

This pattern extends directly to any business that collects free-text customer feedback:

- **Customer support triage** — automatically flag and prioritize negative reviews for faster response
- **Product analytics** — track sentiment trends over app releases to catch regressions early
- **Competitive intelligence** — apply the same pipeline to competitor reviews for benchmarking
- **Voice-of-customer dashboards** — aggregate sentiment breakdowns by feature area or time period for product teams
- **App store reputation monitoring** — set up automated alerts when negative sentiment spikes after a release

The core lesson from this project — that proxy labels (like star ratings) don't always match the underlying signal you're trying to model — is broadly applicable any time a team builds sentiment or classification systems from indirect labels rather than direct human annotation.

## Limitations

- Neutral/mixed-sentiment reviews are classified with substantially lower accuracy than positive or negative reviews, due to the labeling limitation described above
- The dataset is domain-specific to Opay app reviews and may not generalize to other platforms or languages without retraining
- The model was trained primarily on English-language text and may perform poorly on Nigerian Pidgin or heavily code-mixed reviews, which are common in this dataset

## Future Improvements

- Reframe neutral/mixed reviews as a separate multi-label problem (allowing a review to carry both positive and negative tags) rather than a single-label three-way classification
- Collect a smaller set of human-annotated labels (rather than star-derived) to validate and potentially retrain on true sentiment ground truth
- Expand automated testing (pytest) for the API and preprocessing pipeline
- Add model monitoring in production to detect sentiment drift over time as user language or app features evolve
