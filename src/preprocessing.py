import pandas as pd
import json
import gzip

# ----------------------------
# 1. Load Dataset
# ----------------------------
# Path to downloaded file 
file_path = "data/raw/renttherunway_final_data.json.gz"

#Read the gzipped JSON
with gzip.open(file_path, 'rt', encoding='utf-8') as f:
    data = [json.loads(line) for line in f]

# Convert to DataFrame
df = pd.DataFrame(data)

# Quick check
print(df.shape)   # Number of rows and columns
print(df.columns) # All available fields
print(df.head(3)) # First 3 rows

print("Original Shape:", df.shape)

# ----------------------------
# 2. Keep Relevant Columns
# ----------------------------
cols_to_keep = ['review_text', 'rating', 'fit', 'rented for', 
                'body type', 'age', 'weight', 'height', 'category']
df = df[cols_to_keep]

# ----------------------------
# 3. Remove Missing/Empty Reviews
# ----------------------------
df = df[df['review_text'].notnull()]   # remove NaNs
df = df[df['review_text'].str.strip() != ""]  # remove empty strings

# ----------------------------
# 4. Clean Age (convert to numeric & filter realistic range)
# ----------------------------
df['age'] = pd.to_numeric(df['age'], errors='coerce')  # convert to float, set invalid to NaN
df = df[df['age'].notnull()]                           # remove NaNs
df = df[(df['age'] >= 13) & (df['age'] <= 90)]         # keep realistic ages


# ----------------------------
# 5. Clean Weight (keep lbs in reasonable range)
# ----------------------------
def parse_weight(w):
    if pd.isnull(w):
        return None
    try:
        return int(str(w).lower().replace("lbs", "").strip())
    except:
        return None

df['weight'] = df['weight'].apply(parse_weight)
df = df[(df['weight'].isnull()) | ((df['weight'] >= 80) & (df['weight'] <= 350))]

# ----------------------------
# 6. Clean Height (convert to inches)
# ----------------------------
import re # Import Regex

def parse_height(h):
    if pd.isnull(h):
        return None
    match = re.match(r"(\d+)'\s*(\d+)", str(h))
    if match:
        feet, inches = match.groups()
        return int(feet) * 12 + int(inches)
    return None

df['height'] = df['height'].apply(parse_height)
df = df[(df['height'].isnull()) | ((df['height'] >= 48) & (df['height'] <= 78))]  # 4ft to 6'6"

import numpy as np
import pandas as pd

# 1) Make rating numeric
df['rating'] = pd.to_numeric(df['rating'], errors='coerce')

# 2) Drop rows with missing/invalid rating
df = df[df['rating'].notnull()].copy()

# 3) Cast to int for clean comparisons
df['rating'] = df['rating'].astype(int)

# 4) Map to sentiment (auto-detect scale: 1–5 vs 1–10)
scale_max = df['rating'].max()

def map_sentiment(r):
    if scale_max <= 5:
        # 1–5 scale
        if r >= 4: 
            return "Positive"
        elif r == 3:
            return "Neutral"
        else:
            return "Negative"
    else:
        # 1–10 scale (common in this dataset)
        if r >= 8:
            return "Positive"
        elif r >= 6:
            return "Neutral"
        else:
            return "Negative"

df['sentiment'] = df['rating'].apply(map_sentiment)

# (optional) sanity checks
print("Rating scale detected:", scale_max)
print(df['sentiment'].value_counts())
print(df[['rating', 'sentiment']].head())

# ----------------------------
# 8. Final Clean-Up
# ----------------------------
df.reset_index(drop=True, inplace=True)
print("Cleaned Shape:", df.shape)

# ----------------------------
# 9. Save Clean Data
# ----------------------------
df.to_csv("renttherunway_clean.csv", index=False)
print(" Clean dataset saved as 'rent_the_runway_Clean.csv")