FROM python:3.10-slim

WORKDIR /app

COPY requirements/mlflow.txt .
COPY mlflow/entrypoint.sh /entrypoint.sh

RUN apt-get update && apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/* && \
    pip install --no-cache-dir -r mlflow.txt && \
    chmod +x /entrypoint.sh

EXPOSE 5000

ENTRYPOINT ["/entrypoint.sh"]
