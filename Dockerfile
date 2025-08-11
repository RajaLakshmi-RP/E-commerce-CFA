# --- Base image
FROM python:3.11-slim

# --- System settings
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8080

# Optional: install useful libs (helps SciPy/NumPy performance)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# --- Workdir
WORKDIR /app

# --- Install Python deps first
COPY api/requirements.txt ./requirements.txt
RUN python -m pip install --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# --- Copy app code
COPY . .

# --- Create non-root user
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

# --- Expose & healthcheck
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
  CMD curl -fsS http://127.0.0.1:${PORT}/health || exit 1

# --- Start server (entry point is api/main.py)
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8080"]
