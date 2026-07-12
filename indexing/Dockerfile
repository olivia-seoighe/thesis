FROM python:3.12-slim
WORKDIR /app

RUN pip install --upgrade pip

COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy the full indexing package so relative imports work
COPY . ./indexing/

# Run from /app so "from indexing.X import Y" resolves correctly
CMD ["python", "indexing/main.py"]
