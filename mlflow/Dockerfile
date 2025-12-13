FROM python:3.10-slim

WORKDIR /app

COPY requirements/mlflow.txt .

RUN apt-get update && apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/* && \
    pip install --no-cache-dir -r mlflow.txt

EXPOSE 5000

CMD mlflow server \
    --host 0.0.0.0 \
    --port 5000 \
    --backend-store-uri ${DATABASE_URL} \
    --default-artifact-root s3://${S3_BUCKET_MLFLOW}/artifacts
