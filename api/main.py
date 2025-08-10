# api/main.py
from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Optional
import os
import numpy as np
import joblib
import sklearn
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

# ---------------- Paths / Model ----------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "sentiment_pipeline.joblib"
pipe = joblib.load(MODEL_PATH)

# --------------- Optional OpenAI ---------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
USE_OPENAI_EXPLANATION = os.getenv("USE_OPENAI_EXPLANATION", "1") == "1"
openai_client = None
if OPENAI_API_KEY:
    try:
        from openai import OpenAI  # pip install openai
        openai_client = OpenAI(api_key=OPENAI_API_KEY)
    except Exception:
        openai_client = None  # fallback will be used

# ---------------- FastAPI app ------------------
app = FastAPI(title="Product Review Sentiment + Reasoning")

# ----------------- Schemas ---------------------
class ReviewIn(BaseModel):
    review_text: str

class PredictOut(BaseModel):
    sentiment: str
    probability: float
    reason: str            # concise one-liner
    explanation: str       # standalone paragraph (independent of `reason`)
    rephrase: str

# ---------------- Heuristics -------------------
NEGATIVE_HINTS = {
    "broke","broken","tear","ripped","stain","stained","hole","defect",
    "cheap","itchy","uncomfortable","painful","worst","refund","return",
    "faulty","damaged","rip","poor","smell","smelly","loose","tight",
    "stuck","zipper","zip","bleed","shrunk","shrink","pilled","pill",
    "dirty","scratch","scratched","late","missing","broken zipper","bad",
    "pale","light","faded","discolor"
}

ASPECT_KEYWORDS: Dict[str, List[str]] = {
    "defect":    ["broke","broken","tear","ripped","hole","defect","zipper","stuck","button","zip"],
    "quality":   ["quality","cheap","thin","thick","fabric","material","stitch","pilled","pill"],
    "fit":       ["fit","tight","loose","small","large","sizing","size up","size down"],
    "comfort":   ["itchy","uncomfortable","scratchy","heavy","painful"],
    "color":     ["color","dye","bleed","faded","pale","light","discolor"],
    "smell":     ["smell","smelly","odor"],
    "logistics": ["late","missing","return","refund","exchange"],
    "price":     ["price","expensive","overpriced","cheap"],
}

EXPLANATION_TEMPLATES = {
    "Positive": "The language is appreciative and highlights favorable aspects, matching a positive overall experience.",
    "Neutral":  "The wording is balanced or factual with limited emotional cues, so the overall tone is neutral.",
    "Negative": "The text contains dissatisfaction markers and issue-oriented wording that indicate a negative experience.",
}

# Keep the short reason clean (no filler words)
STOPWORDS = {
    "the","a","an","and","or","but","to","of","in","on","for","with","as","at","by",
    "is","was","are","were","be","been","being","it","this","that","these","those",
    "very","really","so","just","too","than","then","there","here","i","we","you",
    "my","our","their","me","us","they","he","she","him","her","them",
    # domain-generic words to hide in 'reason'
    "dress","item","product","rent","rental","order","material","fabric","color","quality"
}

# ----------------- Utils -----------------------
def contains_negative_hint(text: str) -> List[str]:
    t = text.lower()
    return sorted({w for w in NEGATIVE_HINTS if w in t})

def softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x, axis=1, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=1, keepdims=True)

def predict_with_proba(texts: List[str]):
    clf = pipe.named_steps.get("clf")
    X = pipe.named_steps["tfidf"].transform(texts) if "tfidf" in pipe.named_steps else pipe[:-1].transform(texts)
    if hasattr(clf, "predict_proba"):
        probs = clf.predict_proba(X)
        classes = clf.classes_
        return classes, probs
    scores = clf.decision_function(X)
    if scores.ndim == 1:
        scores = np.stack([-scores, scores], axis=1)
    probs = softmax(scores)
    classes = clf.classes_
    return classes, probs

def extract_aspects(text: str) -> List[str]:
    t = text.lower()
    found = []
    for aspect, kws in ASPECT_KEYWORDS.items():
        if any(kw in t for kw in kws):
            found.append(aspect)
    order = ["defect","quality","fit","comfort","color","smell","logistics","price"]
    return [a for a in order if a in set(found)]

def top_negative_terms_word_only(review: str, top_k: int = 3) -> List[str]:
    """
    Return top contributing *word* features (unigrams only) for the Negative class
    that are present in the review and are not stopwords or generic terms.
    Works when tfidf is FeatureUnion([("w", word_vec), ("c", char_vec)]).
    """
    try:
        clf = pipe.named_steps["clf"]
        tfidf = pipe.named_steps["tfidf"]  # FeatureUnion
        if not hasattr(tfidf, "transformer_list") or len(tfidf.transformer_list) == 0:
            return []

        # Pull the word vectorizer (assumed first in the union)
        name_w, vec_w = tfidf.transformer_list[0]
        vocab = np.array(vec_w.get_feature_names_out())
        w_dim = len(vocab)

        # Transform with the union (so indices align to coef_ space)
        X_full = tfidf.transform([review])

        if not hasattr(clf, "coef_"):
            return []
        classes = clf.classes_.tolist()
        if "Negative" not in classes:
            return []
        neg_idx = classes.index("Negative")
        neg_coefs = clf.coef_[neg_idx]

        # Indices present in this document (union space)
        present_idx = X_full.nonzero()[1]
        # Keep only word block (first w_dim)
        present_word_idx = present_idx[present_idx < w_dim]
        if present_word_idx.size == 0:
            return []

        tokens = vocab[present_word_idx]
        mask = [
            ((" " not in tok) and tok.isascii() and tok.isalpha() and len(tok) >= 3
             and tok.lower() not in STOPWORDS)
            for tok in tokens
        ]
        if not any(mask):
            return []

        filtered_idx = present_word_idx[np.array(mask, dtype=bool)]
        filtered_tokens = vocab[filtered_idx]

        # Rank by coef weight (desc), keep positive contributors
        weights = neg_coefs[filtered_idx]
        order = np.argsort(-weights)

        ordered = []
        for i in order:
            if weights[i] <= 0:
                continue
            term = filtered_tokens[i].strip().replace("_", " ")
            if term and term not in ordered:
                ordered.append(term)
            if len(ordered) >= top_k:
                break
        return ordered
    except Exception:
        return []

def build_paragraph_explanation(review: str, sentiment: str, prob: float,
                                reason_terms: List[str], hits: List[str]) -> str:
    """
    Standalone, readable paragraph (does NOT include the short `reason` string).
    """
    base = EXPLANATION_TEMPLATES.get(sentiment, "")
    aspects = extract_aspects(review)

    parts = [base]
    if sentiment == "Negative" and aspects:
        parts.append(" Key areas mentioned include: " + ", ".join(aspects) + ".")
    elif aspects:
        parts.append(" Noted aspects: " + ", ".join(aspects) + ".")

    evidence_bits = []
    if reason_terms: evidence_bits.append("model-weighted cues → " + ", ".join(reason_terms))
    if hits:         evidence_bits.append("issue keywords → " + ", ".join(hits))
    if evidence_bits:
        parts.append(" Evidence: " + "; ".join(evidence_bits) + ".")

    parts.append(f" Confidence ≈ {(prob*100):.1f}%.")
    return "".join(parts)

def try_llm_explanation(review: str, sentiment: str, prob: float,
                        aspects: List[str], evidence: List[str]) -> Optional[str]:
    """
    If OPENAI_API_KEY is set and USE_OPENAI_EXPLANATION=1, generate a 2–3 sentence
    natural explanation. Otherwise return None.
    """
    if not (openai_client and USE_OPENAI_EXPLANATION):
        return None
    try:
        sys = ("You generate concise, neutral explanations for customer review sentiment. "
               "Write 2–3 sentences in plain English, grounded in the provided aspects/evidence. "
               "Do not repeat any rephrasing; focus on explanation only.")
        user = (
            f"Review: {review}\n"
            f"Predicted sentiment: {sentiment} (confidence {prob:.2%})\n"
            f"Aspects: {', '.join(aspects) if aspects else '—'}\n"
            f"Evidence: {', '.join(evidence) if evidence else '—'}\n"
            "Explain briefly why this sentiment is likely."
        )
        resp = openai_client.responses.create(
            model="gpt-4o-mini",
            input=[{"role": "system", "content": sys},
                   {"role": "user", "content": user}],
        )
        text = resp.output_text.strip()
        return text or None
    except Exception:
        return None

def soft_rephrase(review: str, sentiment: str) -> str:
    if sentiment != "Negative":
        return review
    return ("Thank you for the feedback. I experienced issues with this item and it didn’t fully meet my expectations. "
            "I appreciate the design and would value improvements to quality and fit.")

# ------------------- Routes --------------------
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

    # Predict + probability
    classes, probs = predict_with_proba([review])
    probs_row = probs[0]
    idx = int(np.argmax(probs_row))
    sentiment = str(classes[idx])
    prob = float(probs_row[idx])

    # Nudge for clear defect terms (helps short complaints)
    hits = contains_negative_hint(review)
    if hits and not (sentiment == "Positive" and prob >= 0.80):
        sentiment = "Negative"
        prob = max(prob, 0.80)

    # --- Short reason (clean word features only) ---
    reason_terms = top_negative_terms_word_only(review, top_k=3)
    if sentiment == "Negative":
        if reason_terms:
            reason = f"The review specifically mentions {', '.join(reason_terms)}, which indicate dissatisfaction."
        elif hits:
            reason = f"The review contains issue keywords: {', '.join(hits)}."
        else:
            reason = "Wording suggests dissatisfaction with the item."
    else:
        reason = "The wording and tone align with the predicted sentiment."

    # --- Long explanation (independent of `reason`) ---
    aspects = extract_aspects(review)
    evidence_list = []
    if reason_terms: evidence_list.append("model cues: " + ", ".join(reason_terms))
    if hits:         evidence_list.append("keywords: " + ", ".join(hits))

    explanation = build_paragraph_explanation(review, sentiment, prob, reason_terms, hits)

    # Try LLM explanation (optional)
    llm_exp = try_llm_explanation(review, sentiment, prob, aspects, evidence_list)
    if llm_exp:
        explanation = llm_exp

    # Rephrase (for negatives)
    rephrased = soft_rephrase(review, sentiment)

    return PredictOut(
        sentiment=sentiment,
        probability=round(prob, 4),
        reason=reason,
        explanation=explanation,
        rephrase=rephrased
    )

# ------------------- CORS ----------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten to ["http://localhost:5173"] for prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
