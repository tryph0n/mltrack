FROM python:3.10-slim

WORKDIR /app

COPY requirements/streamlit.txt .

RUN apt-get update && apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/* && \
    pip install --no-cache-dir -r streamlit.txt

COPY streamlit/app.py .
COPY streamlit/data.py .

EXPOSE 8501

CMD ["sh", "-c", "streamlit run app.py --server.port=8501 --server.address=0.0.0.0"]
