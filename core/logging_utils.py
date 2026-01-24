from __future__ import annotations

import logging
import os
from datetime import datetime, date


def setup_logging(log_root: str, log_level: str) -> str:
    today = date.today().isoformat()
    log_dir = os.path.join(log_root, today)
    os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%H%M%S")
    log_path = os.path.join(log_dir, f"run_{timestamp}.log")

    logger = logging.getLogger()
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    if logger.handlers:
        logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logger.level)
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logger.level)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    logger.info("Logging initialized. Log file: %s", log_path)

    return log_dir
