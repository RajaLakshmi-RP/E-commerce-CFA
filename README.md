# Product Review Sentiment Analysis – Backend

This is the backend for a sentiment analysis system that takes a product review, detects whether it’s positive, neutral, or negative, explains the reasoning, and rewrites negative reviews in a brand-friendly way.

Currently, this repository contains only the backend code. Frontend and cloud deployment will be added later.

---

### Tech Stack

* **Python 3.11**
* **FastAPI** for the API
* **scikit-learn** for the machine learning model
* **pandas** for data processing

---

### Project Structure
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
  "reason": "Detected negative keywords: broke, zipper.",
  "rephrase": "I had some issues with this item, and it didn’t fully meet my expectations."
}