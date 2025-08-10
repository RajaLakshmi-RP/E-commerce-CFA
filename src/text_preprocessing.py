from __future__ import annotations
import re
import pandas as pd

RE_HTML = re.compile(r"<.*?>")
RE_NON_WORD = re.compile(r"[^a-zA-Z0-9\s]")

def basic_clean(text: str) -> str:
    if not isinstance(text, str):
        return ""
    t = text.lower()
    t = RE_HTML.sub(" ", t)
    t = RE_NON_WORD.sub(" ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

def apply_text_clean(df: pd.DataFrame, text_col: str = "review_text") -> pd.DataFrame:
    out = df.copy()
    out[text_col] = out[text_col].astype(str).apply(basic_clean)
    # Drop empties after cleaning
    out = out[out[text_col].str.strip() != ""]
    out = out.reset_index(drop=True)
    return out

if __name__ == "__main__":
    path_in  = "data/processed/renttherunway_clean.csv"
    path_out = "data/processed/renttherunway_clean_text.csv"
    df = pd.read_csv(path_in)
    df = apply_text_clean(df, "review_text")
    df.to_csv(path_out, index=False)
    print(f"Saved: {path_out} -> {df.shape}")
    
if __name__ == "__main__":
    path_in  = "data/processed/renttherunway_clean.csv"
    path_out = "data/processed/renttherunway_clean_text.csv"
    df = pd.read_csv(path_in)
    df = apply_text_clean(df, "review_text")
    df.to_csv(path_out, index=False)
    print(f"Saved: {path_out} -> {df.shape}")