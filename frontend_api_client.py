"""Lightweight FastAPI client used by the Streamlit frontend."""

import os
import requests

BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000").rstrip("/")


class BackendAPIError(RuntimeError):
    pass


def _unwrap(response: requests.Response):
    response.raise_for_status()
    data = response.json()
    if isinstance(data, dict) and data.get("ok") is False:
        error = data.get("error") or {}
        message = error.get("message") or "Backend request failed"
        detail = error.get("detail")
        raise BackendAPIError(f"{message}: {detail}" if detail else message)
    return data


def backend_health():
    return _unwrap(requests.get(f"{BACKEND_URL}/health", timeout=10))


def backend_chat(message: str, history=None, use_model=True):
    payload = {
        "message": message,
        "history": history or [],
        "use_model": use_model,
    }
    return _unwrap(requests.post(f"{BACKEND_URL}/api/chat", json=payload, timeout=180))


def backend_analyze_log(text: str):
    return _unwrap(requests.post(f"{BACKEND_URL}/api/log/analyze", json={"text": text}, timeout=120))


def backend_explain_flow(text: str):
    return _unwrap(requests.post(f"{BACKEND_URL}/api/flow/explain", json={"text": text}, timeout=120))


def backend_extract_iocs(text: str):
    return _unwrap(requests.post(f"{BACKEND_URL}/api/ioc/extract", json={"text": text}, timeout=60))


def backend_attack_chain(text: str):
    return _unwrap(requests.post(f"{BACKEND_URL}/api/attack-chain/analyze", json={"text": text}, timeout=60))


def backend_soar_generate(requirement: str):
    return _unwrap(requests.post(f"{BACKEND_URL}/api/soar/generate", json={"requirement": requirement}, timeout=60))


def backend_soar_simulate(yaml_text: str):
    return _unwrap(requests.post(f"{BACKEND_URL}/api/soar/simulate", json={"yaml_text": yaml_text}, timeout=60))
