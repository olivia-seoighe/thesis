import os

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI

load_dotenv()

from retrieval.utils.logging_config import get_logger, setup_logging

setup_logging()
logger = get_logger(__name__)

app = FastAPI(title="Code RAG Search API")

from retrieval.routes import router

app.include_router(router)

logger.info("Starting Code RAG Search API")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)