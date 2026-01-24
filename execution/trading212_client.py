from __future__ import annotations

import base64
import logging
from typing import Any, Dict, Optional

import requests

logger = logging.getLogger(__name__)


class Trading212Error(RuntimeError):
    pass


class Trading212Client:
    def __init__(
        self,
        *,
        api_key: str,
        api_secret: str,
        base_url: str,
        timeout_seconds: float = 30.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": self._build_auth_header(api_key, api_secret),
            "Accept": "application/json",
        })
        logger.info("Trading212 client configured for %s", self.base_url)

    def get_instruments(self) -> Any:
        return self._request("GET", "/api/v0/equity/metadata/instruments")

    def get_positions(self) -> Any:
        return self._request("GET", "/api/v0/equity/positions")

    def get_account_summary(self) -> Any:
        return self._request("GET", "/api/v0/equity/account/summary")

    def place_market_order(
        self,
        *,
        ticker: str,
        quantity: float,
        extended_hours: bool = False,
    ) -> Any:
        payload = {
            "ticker": ticker,
            "quantity": quantity,
            "extendedHours": extended_hours,
        }
        logger.info(
            "Placing market order: ticker=%s quantity=%s extendedHours=%s",
            ticker,
            quantity,
            extended_hours,
        )
        return self._request("POST", "/api/v0/equity/orders/market", json=payload)

    def _request(
        self,
        method: str,
        path: str,
        *,
        params: Optional[Dict[str, Any]] = None,
        json: Optional[Dict[str, Any]] = None,
    ) -> Any:
        url = path
        if not path.startswith("http"):
            url = f"{self.base_url}{path}"

        try:
            response = self.session.request(
                method,
                url,
                params=params,
                json=json,
                timeout=self.timeout_seconds,
            )
        except requests.RequestException as exc:
            logger.error("Trading212 request failed: %s %s (%s)", method, url, exc)
            raise Trading212Error(str(exc)) from exc

        if response.status_code >= 400:
            message = response.text.strip()
            logger.error(
                "Trading212 API error: %s %s status=%s body=%s",
                method,
                url,
                response.status_code,
                message,
            )
            raise Trading212Error(f"HTTP {response.status_code} for {url}: {message}")

        if response.content:
            try:
                return response.json()
            except ValueError as exc:
                logger.error("Trading212 response was not valid JSON: %s", exc)
                raise Trading212Error("Invalid JSON response from Trading212") from exc
        return None

    @staticmethod
    def _build_auth_header(api_key: str, api_secret: str) -> str:
        credentials = f"{api_key}:{api_secret}".encode("utf-8")
        encoded = base64.b64encode(credentials).decode("ascii")
        return f"Basic {encoded}"
