# ChurnIQ prediction API. Build after training (needs models/churn_model.joblib):
#   docker build -t churniq-api .
#   docker run -p 8000:8000 churniq-api
FROM python:3.11-slim

# libgomp1: OpenMP runtime required by LightGBM
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir .

COPY models ./models

EXPOSE 8000
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "churniq.api:app", "--host", "0.0.0.0", "--port", "8000"]
