# Product Review Sentiment + Reasoning

An **explainable machine learning** system that classifies customer reviews as **Positive, Neutral, or Negative**, explains the reasoning behind the classification, and generates **brand-friendly rephrases** for customer communication.

---

### Project Structure
```
├─ api/
│ └─ main.py # FastAPI app
├─ src/
│ ├─ preprocessing.py # Raw → clean CSV
│ ├─ text_preprocessing.py # Extra text cleaning
│ └─ sentiment_model.py # Train & save model
├─ data/
│ ├─ raw/ # Place raw dataset here
│ └─ processed/ # Generated cleaned data
├─ models/ # Trained model files
└─ README.md
```

## Overview

This project goes beyond basic sentiment analysis by providing:

- **Classification**: Positive / Neutral / Negative sentiment.
- **Explainability**: Model-weighted cues, issue keywords, and aspect detection.
- **Rephrasing**: Generates polite, brand-aligned responses with tone and length control.
- **LLM-Optional**: Fully functional without LLMs; enhanced explanations and rephrases when API key is available.

---

## Tech Stack

- **Python 3.11**
- **FastAPI** for serving the API
- **scikit-learn** for the ML pipeline
- **pandas** for data handling
- **React + Vite** for the frontend
- **Optional:** OpenAI API for enhanced explanations/rephrases

---
### Setup Instructions

**1. Create a virtual environment**

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

**2. Preparing the Data**

```bash
Place the raw file in:
data/raw/renttherunway_final_data.json.gz

Run preprocessing:
python src/preprocessing.py
python src/text_preprocessing.py

This will create:
data/processed/renttherunway_clean.csv
data/processed/renttherunway_clean_text.csv

*** 3. Training the Model***
Run: python src/sentiment_model.py

This will save the trained model to:
models/sentiment_pipeline.joblib

Running the API
Start the server:

uvicorn api.main:app --host 0.0.0.0 --port 8080
Open the interactive docs in your browser:
http://127.0.0.1:8080/docs

Example API Usage
POST /predict

JSON

{
  "review_text": "The zipper broke after one use."
}
Response

JSON

{
  "sentiment": "Negative",
  "probability": 0.89,

## Key Features

1. **Robust Text Preprocessing**  
   - Lowercasing, HTML tag removal, special character filtering.  
   - Drop empty or invalid reviews.

2. **Domain-Specific Heuristics**  
   - Detects defect-related keywords (e.g., "zipper", "ripped").  
   - Aspect categories: defect, quality, fit, comfort, color, smell, logistics, price.  
   - Heuristic boost for critical issues to improve recall.

3. **Model Architecture**  
   - **FeatureUnion** TF-IDF:  
     - Word n-grams (1–2) for semantic meaning.  
     - Char n-grams (3–4) for handling typos and short words.  
   - Classifier: **LinearSVC** with balanced class weights.  
   - Optimized for CPU deployment.

4. **Explainability**  
   - Extracts top negative terms from model coefficients.  
   - Highlights issue keywords and relevant aspects.  
   - Generates short "reason" and a longer "explanation".

5. **Rephrasing**  
   - Template-based rephrasing with configurable tone and length.  
   - Optional LLM refinement for negative reviews.

6. **API Endpoints**  
   - `/health` – Service health check  
   - `/env` – Environment info (e.g., sklearn version)  
   - `/predict` – Returns sentiment, probability, reason, explanation, rephrase  
   - `/rephrase` – Returns rephrase for a given review with tone/length controls  

---

## Setup Instructions

### 1. Create Virtual Environment and Install Dependencies
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Data Preparation
Place the raw dataset at:

```
bash
data/raw/renttherunway_final_data.json.gz
```

Run preprocessing:
```bash
python src/preprocessing.py
python src/text_preprocessing.py

```

```bash

data/processed/renttherunway_clean.csv
data/processed/renttherunway_clean_text.csv

```
### 3. Train the Model
```
bash

python src/sentiment_model.py
Saves model to:
```
```bash
models/sentiment_pipeline.joblib
```
### 4. Run the API
```
bash

uvicorn api.main:app --host 0.0.0.0 --port 8080
```
Interactive API docs:
```
arduino

http://127.0.0.1:8080/docs
```

Example API Usage
POST /predict
```
Request:

json
{
  "review_text": "The zipper broke after one use."
}
```
```
Response:

json

{
  "sentiment": "Negative",
  "probability": 0.89,
  "reason": "The review specifically mentions broke and zipper, which indicate dissatisfaction.",
  "explanation": "The text contains dissatisfaction markers... Confidence ≈ 89.0%.",
  "rephrase": "Thanks for the review. We understand there were hardware issues (e.g., zipper/button). We’ll use this to improve."
}
```
```
Environment Variables
Variable	Description
OPENAI_API_KEY	(Optional) API key for enhanced LLM-based explanations
USE_OPENAI_EXPLANATION	Set 1 to enable LLM explanations if key is present (default)
```
```
Deployment
Local: Run with Uvicorn (as shown above)

Containerized: Use provided Dockerfile

Cloud: Compatible with GCP Cloud Run, AWS ECS, Azure Container Apps
```
```
Frontend can be deployed to static hosting (e.g., Netlify, Vercel, Cloud Storage + CDN)

Limitations & Future Improvements
Approximate probabilities for SVC — consider probability calibration.

Expand aspect categories and domain keyword lists.

Add multilingual support.

Implement feedback loop to learn from user-edited rephrases.
```
```
License
This project is licensed under the MIT License.
```
```

  "reason": "Detected negative keywords: broke, zipper.",
  "rephrase": "I had some issues with this item, and it didn’t fully meet my expectations."
}
