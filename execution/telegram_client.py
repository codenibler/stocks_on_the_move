from __future__ import annotations

import glob
import logging
import os
import time
from typing import Iterable, Optional

import requests

logger = logging.getLogger(__name__)


class TelegramError(RuntimeError):
    pass


class TelegramClient:
    def __init__(self, *, api_token: str, timeout_seconds: float = 30.0) -> None:
        self.base_url = f"https://api.telegram.org/bot{api_token}"
        self.timeout_seconds = timeout_seconds
        self.session = requests.Session()

    def send_message(self, chat_id: str, text: str) -> None:
        self._post("sendMessage", data={"chat_id": chat_id, "text": text, "parse_mode": "HTML"})

    def send_photo(self, chat_id: str, photo_path: str, caption: Optional[str] = None) -> None:
        data = {"chat_id": chat_id}
        if caption:
            data["caption"] = caption
        with open(photo_path, "rb") as handle:
            self._post("sendPhoto", data=data, files={"photo": handle})

    def send_document(self, chat_id: str, document_path: str, caption: Optional[str] = None) -> None:
        data = {"chat_id": chat_id}
        if caption:
            data["caption"] = caption
        with open(document_path, "rb") as handle:
            self._post("sendDocument", data=data, files={"document": handle})

    def _post(self, method: str, *, data: dict, files: Optional[dict] = None) -> None:
        url = f"{self.base_url}/{method}"
        try:
            response = self.session.post(
                url,
                data=data,
                files=files,
                timeout=self.timeout_seconds,
            )
        except requests.RequestException as exc:
            raise TelegramError(str(exc)) from exc
        if response.status_code >= 400:
            raise TelegramError(f"HTTP {response.status_code} for {method}: {response.text.strip()}")


def send_rebalance_report(
    *,
    api_token: str,
    chat_id: str,
    report_path: str,
    pages_dir: str,
    send_delay_seconds: float = 1.0,
    caption: Optional[str] = None,
    message_text: Optional[str] = None,
    message_blocks: Optional[list[dict]] = None,
    send_pages: bool = False,
) -> None:
    client = TelegramClient(api_token=api_token)
    if message_blocks:
        for block in message_blocks:
            if block.get("type") == "text":
                client.send_message(chat_id, block.get("text", ""))
            elif block.get("type") == "photo":
                client.send_photo(chat_id, block.get("path", ""), caption=block.get("caption"))
            elif block.get("type") == "document":
                client.send_document(chat_id, block.get("path", ""), caption=block.get("caption"))
            time.sleep(send_delay_seconds)
    elif message_text:
        client.send_message(chat_id, message_text)
        time.sleep(send_delay_seconds)
    if send_pages:
        page_paths = sorted(glob.glob(os.path.join(pages_dir, "page_*.png")))
        if page_paths:
            total_pages = len(page_paths)
            for idx, page_path in enumerate(page_paths, start=1):
                page_caption = None
                if idx == 1:
                    page_caption = caption or f"Rebalance report ({total_pages} pages)"
                client.send_photo(chat_id, page_path, caption=page_caption)
                time.sleep(send_delay_seconds)
        else:
            logger.warning("No report page images found in %s", pages_dir)

    if os.path.isfile(report_path):
        client.send_document(chat_id, report_path, caption="Rebalance report (PDF)")
    else:
        logger.warning("Report PDF not found at %s", report_path)
