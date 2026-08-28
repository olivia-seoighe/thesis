import logging
import os
import sys
import uuid
from contextvars import ContextVar
from pathlib import Path
from typing import Optional

request_id_var: ContextVar[Optional[str]] = ContextVar("request_id", default=None)


def generate_request_id() -> str:
    return str(uuid.uuid4())


def set_request_id(request_id: str) -> None:
    request_id_var.set(request_id)


def get_request_id() -> Optional[str]:
    return request_id_var.get()


class ServiceAdapter(logging.LoggerAdapter):
    def process(self, msg, kwargs):
        extra = kwargs.get("extra", {})
        service_info = {
            "service": {
                "name": "retrieval",
                "version": os.getenv("SERVICE_VERSION", "0.1.0"),
                "environment": os.getenv("ENVIRONMENT", "development"),
            }
        }
        request_id = get_request_id()
        if request_id:
            service_info["trace"] = {"id": request_id}
        extra.update(service_info)
        kwargs["extra"] = extra
        return msg, kwargs


def setup_logging() -> logging.Logger:
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    log_file_path = os.getenv("LOG_FILE_PATH", "logs/retrieval.log")

    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")

    root = logging.getLogger()
    root.setLevel(getattr(logging, log_level, logging.INFO))

    for handler in root.handlers[:]:
        root.removeHandler(handler)

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    root.addHandler(handler)

    file_path = Path(log_file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(file_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)

    logging.getLogger("asyncio").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    return root


def get_logger(name: str) -> ServiceAdapter:
    return ServiceAdapter(logging.getLogger(name), {})