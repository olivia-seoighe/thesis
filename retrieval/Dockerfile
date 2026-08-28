FROM python:3.12-slim
WORKDIR /app

RUN pip install --upgrade pip

COPY requirements.txt ./
RUN pip install -r requirements.txt

COPY . ./retrieval/

# retrieval/ modules use absolute package imports from retrieval.*
ENV PYTHONPATH=/app

EXPOSE 8000
CMD ["python", "-m", "uvicorn", "retrieval.main:app", "--host", "0.0.0.0", "--port", "8000"]
