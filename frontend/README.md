CORS: the API is configured to allow all origins in dev. For production, restrict to your frontend origin.

2) Frontend (Vite)
bash
Copy
Edit
cd frontend
npm install
# Point the frontend to the API (create .env at frontend/):
# VITE_API_BASE_URL=http://127.0.0.1:8080
npm run dev
# Vite will start at http://localhost:5173 (or 5174/5175 if busy)

Environment Variables
Place these in .env (never commit it):

Backend

OPENAI_API_KEY – optional; enables LLM-style explanation/polish

USE_OPENAI_EXPLANATION=1 – set to 0 to disable even if key exists

Frontend (frontend/.env)

VITE_API_BASE_URL=http://127.0.0.1:8080

API Endpoints
GET /health → { "status": "ok" }

GET /env → { "sklearn": "<version>" }

POST /predict

Body: { "review_text": "..." }

Returns: { sentiment, probability, reason, explanation, rephrase }

POST /rephrase

Body: { review_text, tone="neutral", length="medium", sentiment? }

Returns: { rephrase }

GET /docs – Swagger UI

Datasets & Git LFS
Large CSVs in data/processed/*.csv are tracked with Git LFS.
If cloning fresh:

bash
Copy
Edit
git lfs install
git lfs pull

Troubleshooting
Blank page after Analyze
Ensure frontend env points to backend:
frontend/.env → VITE_API_BASE_URL=http://127.0.0.1:8080
Restart npm run dev after changing .env.

CORS error in browser
Confirm backend is running and CORS is permissive (dev mode allows *).

Port already in use

Vite: it will auto-increment (5173 → 5174/5175).

Uvicorn: change --port 8081 or kill the old process.

OpenAI errors

Remove or rotate your key if leaked.

Set USE_OPENAI_EXPLANATION=0 to disable.

🛡 Security & Hygiene
Secrets live only in .env (repo ignores it).

.venv/ & node_modules/ are ignored.

Secret scanning clean; if flagged on push, rotate key and scrub history.

Future Ideas
Model training notebook + evaluation

More tones/styles for rephrase

User auth + saved analyses

Docker Compose for one-command dev

License
MIT (or your choice). Update this section if needed.

yaml
Copy
Edit

---

# Optional `frontend/README.md` (replace the template)

```markdown
# Frontend (React + Vite)

Dev UI for the Review Sentiment + Reasoning app.

## Setup

```bash
cd frontend
npm install
# Create .env with the API URL:
# VITE_API_BASE_URL=http://127.0.0.1:8080
npm run dev
App runs at http://localhost:5173 (auto-increments if busy).

Make sure the backend (FastAPI) is running.

Scripts
npm run dev – start Vite dev server

npm run build – production build

npm run preview – preview the build

Notes
Styles in src/styles.css

Components in src/components/

API calls in src/api.js





