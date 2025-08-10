# src/sentiment_model.py
from __future__ import annotations
from pathlib import Path
import argparse
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.svm import LinearSVC

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "renttherunway_clean_text.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "sentiment_pipeline.joblib"
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

def build_pipeline() -> Pipeline:
    # Lighter/faster settings; you can relax later after verifying
    word_tfidf = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),      # trim to bigrams for speed
        min_df=5,                # ignore very rare tokens
        max_features=120_000,
        sublinear_tf=True
    )
    char_tfidf = TfidfVectorizer(
        analyzer="char",
        ngram_range=(3, 4),      # slightly smaller span
        min_df=5,
        max_features=60_000,
        sublinear_tf=True
    )

    feats = FeatureUnion(
        [("w", word_tfidf), ("c", char_tfidf)],
        n_jobs=-1                 # parallelize feature extraction branches
    )

    pipe = Pipeline([
        ("tfidf", feats),
        ("clf", LinearSVC(class_weight="balanced", random_state=42)),
    ])
    return pipe

def main(sample_frac: float):
    df = pd.read_csv(DATA_PATH)
    if sample_frac and 0 < sample_frac < 1.0:
        df = df.sample(frac=sample_frac, random_state=42).reset_index(drop=True)
        print(f"Using sample: {len(df):,} rows")

    X = df["review_text"].astype(str)
    y = df["sentiment"].astype(str)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipe = build_pipeline()
    pipe.fit(X_tr, y_tr)

    preds = pipe.predict(X_te)
    print("Accuracy:", round(accuracy_score(y_te, preds), 4))
    print(classification_report(y_te, preds))

    joblib.dump(pipe, MODEL_PATH)
    print(f"✅ Saved model -> {MODEL_PATH}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample-frac", type=float, default=0.25,
                    help="Fraction of data to train on for a fast run (0<frac<=1). Set to 1 for full data.")
    args = ap.parse_args()
    main(sample_frac=args.sample_frac)
