FROM python:3.12-slim
WORKDIR /app

RUN pip install --upgrade pip

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

# summaries written to /app/temp — mounted as a Docker volume
RUN mkdir -p /app/temp

EXPOSE 8001
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]
