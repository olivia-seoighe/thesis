FROM python:3.12-slim
WORKDIR /app

RUN pip install --upgrade pip

COPY requirements.txt ./
RUN pip install -r requirements.txt

COPY . ./generation/

# generation/ modules use bare imports (from models import X, etc.)
ENV PYTHONPATH=/app/generation

EXPOSE 8002
CMD ["python", "-m", "uvicorn", "main:app", "--app-dir", "generation", "--host", "0.0.0.0", "--port", "8002"]
