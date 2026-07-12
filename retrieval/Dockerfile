FROM python:3.12-slim
WORKDIR /app

RUN pip install --upgrade pip

COPY requirements.txt ./
RUN pip install -r requirements.txt

COPY . ./retrieval/

# retrieval/ modules use bare imports (from clients.X, from models.X, etc.)
ENV PYTHONPATH=/app/retrieval

EXPOSE 8000
CMD ["python", "-m", "uvicorn", "main:app", "--app-dir", "retrieval", "--host", "0.0.0.0", "--port", "8000"]
