from pathlib import Path
import json
import urllib.request
import urllib.error

CONFIG_PATH = Path("outputs/model_config.json")


def load_model_config():
    if CONFIG_PATH.exists():
        try:
            return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass

    return {
        "mode": "ollama",
        "ollama_url": "http://localhost:11434",
        "model_name": "qwen2.5:3b",
    }


def save_model_config(config: dict):
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")


def call_ollama(prompt: str, model_name: str = "qwen2.5:3b", base_url: str = "http://localhost:11434", timeout: int = 120) -> str:
    base_url = base_url.rstrip("/")
    url = f"{base_url}/api/generate"

    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.3,
            "top_p": 0.9,
        }
    }

    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8", errors="ignore")
            parsed = json.loads(raw)
            return parsed.get("response", "").strip() or raw
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Ollama 连接失败：{exc}") from exc
    except Exception as exc:
        raise RuntimeError(f"模型调用失败：{exc}") from exc
