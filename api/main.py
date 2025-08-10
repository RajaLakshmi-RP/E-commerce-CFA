# api/main.py
from __future__ import annotations
from pathlib import Path
from typing import List
import numpy as np
import joblib
import sklearn
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from pydantic import BaseModel

# ---------- Paths ----------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "sentiment_pipeline.joblib"

# Load trained pipeline
pipe = joblib.load(MODEL_PATH)

app = FastAPI(title="Product Review Sentiment + Reasoning")

# ---------- Schemas ----------
class ReviewIn(BaseModel):
    review_text: str

class PredictOut(BaseModel):
    sentiment: str
    probability: float
    reason: str
    rephrase: str

# ---------- Helpers ----------
NEGATIVE_HINTS = {
    "broke", "broken", "tear", "ripped", "stain", "stained", "hole", "defect",
    "cheap", "itchy", "uncomfortable", "painful", "worst", "refund", "return",
    "faulty", "damaged", "rip", "poor", "smell", "smelly", "loose", "tight",
    "stuck", "zipper", "zip", "bleed", "shrunk", "shrink", "pilled", "pill",
    "dirty", "scratch", "scratched", "late", "missing", "broken zipper"
}

def contains_negative_hint(text: str) -> List[str]:
    t = text.lower()
    hits = [w for w in NEGATIVE_HINTS if w in t]
    return hits

def softmax(x: np.ndarray) -> np.ndarray:
    # stable softmax for 2D scores
    x = x - np.max(x, axis=1, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=1, keepdims=True)

def predict_with_proba(texts: List[str]):
    """Return (classes, probs) for one or more texts, supporting both
    classifiers with predict_proba and those with decision_function."""
    clf = pipe.named_steps.get("clf")
    if hasattr(clf, "predict_proba"):
        probs = clf.predict_proba(pipe.named_steps["tfidf"].transform(texts) if "tfidf" in pipe.named_steps else pipe[:-1].transform(texts))
        classes = clf.classes_
        return classes, probs
    # Fallback: use decision_function and softmax
    X = pipe.named_steps["tfidf"].transform(texts) if "tfidf" in pipe.named_steps else pipe[:-1].transform(texts)
    scores = clf.decision_function(X)
    if scores.ndim == 1:
        scores = np.stack([-scores, scores], axis=1)  # binary case
    probs = softmax(scores)
    classes = clf.classes_
    return classes, probs

def top_negative_terms(review: str, top_k: int = 3) -> List[str]:
    """Heuristic explanation: show tokens present in the review
    that have strong weights toward the Negative class."""
    try:
        clf = pipe.named_steps["clf"]
        vec = pipe.named_steps["tfidf"]
        classes = clf.classes_.tolist()
        if "Negative" not in classes or not hasattr(clf, "coef_"):
            return []
        neg_idx = classes.index("Negative")
        x = vec.transform([review])
        coefs = clf.coef_[neg_idx]
        feats = np.array(vec.get_feature_names_out())
        present_idx = x.nonzero()[1]
        if present_idx.size == 0:
            return []
        present_feats = feats[present_idx]
        present_coefs = coefs[present_idx]
        order = np.argsort(-present_coefs)
        top = [(present_feats[i], present_coefs[i]) for i in order[:top_k] if present_coefs[i] > 0]
        return [w for w, _ in top]
    except Exception:
        # If vectorizer or coef lookup fails (e.g., different pipeline), just return none
        return []

def soft_rephrase(review: str, sentiment: str) -> str:
    if sentiment != "Negative":
        return review
    return (
        "I had some difficulties with this item. "
        "The fit and overall experience didn’t fully meet my expectations, "
        "but I appreciate the design and hope for improvements."
    )

# ---------- Routes ----------
@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/env")
def env():
    return {"sklearn": sklearn.__version__}

@app.post("/predict", response_model=PredictOut)
def predict(inp: ReviewIn):
    review = inp.review_text

    # Model prediction + probability
    classes, probs = predict_with_proba([review])
    probs_row = probs[0]
    idx = int(np.argmax(probs_row))
    sentiment = str(classes[idx])
    prob = float(probs_row[idx])

    # Keyword-assisted nudge for obvious defects (helps very short complaints)
    hits = contains_negative_hint(review)
    if hits and not (sentiment == "Positive" and prob >= 0.80):
        sentiment = "Negative"
        prob = max(prob, 0.80)

    # Reasoning terms
    reason_terms = top_negative_terms(review, top_k=3)
    if sentiment == "Negative":
        if reason_terms:
            reason = f"Detected strongly negative cues: {', '.join(reason_terms)}."
        elif hits:
            reason = f"Detected issue keywords: {', '.join(sorted(set(hits)))}."
        else:
            reason = "Language patterns indicate dissatisfaction with the item."
    else:
        reason = "Overall language and patterns align with the predicted sentiment."

    # Brand-friendly rephrase (only for negatives)
    rephrased = soft_rephrase(review, sentiment)

    return PredictOut(
        sentiment=sentiment,
        probability=round(prob, 4),
        reason=reason,
        rephrase=rephrased
    )
