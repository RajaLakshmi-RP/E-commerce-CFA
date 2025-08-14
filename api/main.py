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
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

#  Load sentiment analysis model
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "sentiment_pipeline.joblib"
pipe = joblib.load(MODEL_PATH)

#  OpenAI Setup
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
USE_OPENAI_EXPLANATION = os.getenv("USE_OPENAI_EXPLANATION", "1") == "1"
openai_client = None
if OPENAI_API_KEY:
    try:
        from openai import OpenAI  # pip install openai
        openai_client = OpenAI(api_key=OPENAI_API_KEY)
    except Exception:
        openai_client = None  # fallback will be used

#  FastAPI app configuration
app = FastAPI(title="Product Review Sentiment + Reasoning")

#  CORS (DEV) (allow front end to call API directly )
# Fast for development: allow all origins so Vite (5173/4/5) can call 127.0.0.1:8080
# IMPORTANT: tighten this for production (e.g., allow_origins=["https://yourdomain"])
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

#  Schemas 
class ReviewIn(BaseModel):
    review_text: str

class PredictOut(BaseModel):
    sentiment: str
    probability: float
    reason: str
    explanation: str
    rephrase: str

class RephraseIn(BaseModel):
    review_text: str
    tone: str | None = "neutral"   # neutral | apologetic | warm | concise | formal
    length: str | None = "medium"  # short | medium | long
    sentiment: str | None = None   # optional: pass predicted sentiment

class RephraseOut(BaseModel):
    rephrase: str

#  Heuristics 
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

STOPWORDS = {
    "the","a","an","and","or","but","to","of","in","on","for","with","as","at","by",
    "is","was","are","were","be","been","being","it","this","that","these","those",
    "very","really","so","just","too","than","then","there","here","i","we","you",
    "my","our","their","me","us","they","he","she","him","her","them",
    # hide generic domain words in 'reason'
    "dress","item","product","rent","rental","order","material","fabric","color","quality"
}

#  Utility Function
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
    # decision_function fallback
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

        name_w, vec_w = tfidf.transformer_list[0]  # word vec assumed first
        vocab = np.array(vec_w.get_feature_names_out())
        w_dim = len(vocab)

        X_full = tfidf.transform([review])

        if not hasattr(clf, "coef_"):
            return []
        classes = clf.classes_.tolist()
        if "Negative" not in classes:
            return []
        neg_idx = classes.index("Negative")
        neg_coefs = clf.coef_[neg_idx]

        present_idx = X_full.nonzero()[1]
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

def _clean_terms(terms: List[str]) -> List[str]:
    out, seen = [], set()
    for t in terms:
        tl = t.lower().strip()
        if not tl or tl in seen:
            continue
        seen.add(tl)
        out.append(tl)
    return out

def _pretty_join(lst: List[str]) -> str:
    if not lst: return ""
    if len(lst) == 1: return lst[0]
    if len(lst) == 2: return f"{lst[0]} and {lst[1]}"
    return f"{', '.join(lst[:-1])}, and {lst[-1]}"

def build_paragraph_explanation(review: str, sentiment: str, prob: float,
                                reason_terms: List[str], hits: List[str]) -> str:
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
        text = (resp.output_text or "").strip()
        return text or None
    except Exception:
        return None

#  Rephrase (smart, tone/length-aware) 
def generate_rephrase(review: str, sentiment: str, aspects: List[str],
                      tone: str = "neutral", length: str = "medium") -> str:
    """
    Non-LLM, brand-friendly rephrase builder.
    tone: neutral | apologetic | warm | concise | formal
    length: short | medium | long
    """
    if sentiment != "Negative":
        base = "Thanks for sharing your experience. "
        if length == "short":
            return base + "We’re glad it worked for you."
        if length == "long":
            return base + "We appreciate your thoughts and will keep them in mind."
        return base + "Appreciate the feedback!"

    intro_by_tone = {
        "neutral":     "Thanks for the review.",
        "apologetic":  "We’re sorry about the trouble you experienced.",
        "warm":        "Thank you for letting us know about your experience.",
        "concise":     "Thanks for your feedback.",
        "formal":      "We appreciate your detailed feedback.",
    }
    close_by_tone = {
        "neutral":     "We will use this to improve.",
        "apologetic":  "We will  address this with our team and improve.",
        "warm":        "We will share this with our team to make things better.",
        "concise":     "We will improve.",
        "formal":      "Your comments will be reviewed to inform product improvements.",
    }

    aspect_map = {
        "defect":    "there were hardware issues (e.g., zipper/button)",
        "color":     "the color did not match expectations",
        "fit":       "the sizing/fit wasn’t right",
        "quality":   "the material/quality didn’t feel right",
        "comfort":   "the item felt uncomfortable",
        "smell":     "there was an odor concern",
        "logistics": "there were order or return issues",
        "price":     "the price didn’t feel justified",
    }

    intro = intro_by_tone.get(tone, intro_by_tone["neutral"])
    close = close_by_tone.get(tone, close_by_tone["neutral"])
    aspect_bits = [aspect_map[a] for a in aspects if a in aspect_map] or ["the item didn’t fully meet expectations"]

    if length == "short":
        body = f" We understand {aspect_bits[0]}."
    elif length == "long":
        joined = "; ".join(aspect_bits[:3])
        body = f" We understand that {joined}. We value your feedback and will look into this."
    else:
        joined = "; ".join(aspect_bits[:2])
        body = f" We understand {joined}."
    return f"{intro} {body} {close}"

def soft_rephrase(review: str, sentiment: str) -> str:
    # default suggestion used by /predict (tone/length controls live under /rephrase)
    aspects = extract_aspects(review)
    return generate_rephrase(review, sentiment, aspects, tone="neutral", length="medium")

# Routes 
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

    # Nudge when clear defect terms appear
    hits = contains_negative_hint(review)
    if hits and not (sentiment == "Positive" and prob >= 0.80):
        sentiment = "Negative"
        prob = max(prob, 0.80)

    # Short reason (clean terms: model + keywords)
    reason_terms = top_negative_terms_word_only(review, top_k=3)
    merged_terms = _clean_terms(reason_terms + hits)[:3]
    if sentiment == "Negative":
        if merged_terms:
            reason = f"The review specifically mentions {_pretty_join(merged_terms)}, which indicate dissatisfaction."
        else:
            reason = "Wording suggests dissatisfaction with the item."
    else:
        reason = "The wording and tone align with the predicted sentiment."

    # Long explanation (template or LLM)
    aspects = extract_aspects(review)
    explanation = build_paragraph_explanation(review, sentiment, prob, reason_terms, hits)
    llm_exp = try_llm_explanation(review, sentiment, prob, aspects, [])
    if llm_exp:
        explanation = llm_exp

    # Suggested rephrase (neutral/medium); UI can regenerate via /rephrase
    rephrased = soft_rephrase(review, sentiment)

    return PredictOut(
        sentiment=sentiment,
        probability=round(prob, 4),
        reason=reason,
        explanation=explanation,
        rephrase=rephrased
    )

@app.post("/rephrase", response_model=RephraseOut)
def rephrase_api(inp: RephraseIn):
    review = inp.review_text
    tone = (inp.tone or "neutral").lower()
    length = (inp.length or "medium").lower()

    # reuse model sentiment if not provided
    snt = inp.sentiment
    if not snt:
        classes, probs = predict_with_proba([review])
        snt = str(classes[int(np.argmax(probs[0]))])

    aspects = extract_aspects(review)
    base = generate_rephrase(review, snt, aspects, tone=tone, length=length)

    #  LLM polish for negative cases
    if openai_client and snt == "Negative":
        try:
            sys = ("Rewrite this into a short, brand-friendly response that acknowledges the issues and sets an improving tone. Avoid fluff.")
            prompt = (f"Review: {review}\n"
                      f"Detected aspects: {', '.join(aspects) or '—'}\n"
                      f"Tone: {tone}; Length: {length}\n"
                      f"Base draft: {base}\n"
                      "Return a single refined sentence or two that keeps the meaning.")
            resp = openai_client.responses.create(
                model="gpt-4o-mini",
                input=[{"role":"system","content":sys},{"role":"user","content":prompt}],
            )
            text = (resp.output_text or "").strip()
            if text:
                return RephraseOut(rephrase=text)
        except Exception:
            pass

    return RephraseOut(rephrase=base)

#  Run directly with: python -m uvicorn api.main:app --reload --host 127.0.0.1 --port 8080
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="127.0.0.1", port=8080, reload=True)
