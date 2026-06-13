# Backend Guide

The FastAPI backend is optional for the local demo because the Streamlit frontend can fall back to local analysis functions. Running the backend is recommended when reviewing API design or integrating an external model provider.

## Start

```powershell
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```

## URLs

| Purpose | URL |
|---|---|
| Health | `http://127.0.0.1:8000/health` |
| Swagger UI | `http://127.0.0.1:8000/docs` |
| OpenAPI JSON | `http://127.0.0.1:8000/openapi.json` |

## Main Endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/health` | Backend status |
| `POST` | `/api/chat` | RAG-assisted assistant |
| `POST` | `/api/log/analyze` | Log analysis + IOC + attack chain |
| `POST` | `/api/flow/explain` | Traffic summary explanation |
| `POST` | `/api/ioc/extract` | IOC extraction |
| `POST` | `/api/attack-chain/analyze` | Attack-chain reconstruction |
| `POST` | `/api/soar/generate` | SOAR YAML generation |
| `POST` | `/api/soar/simulate` | Simulated playbook execution |

See [docs/API.md](docs/API.md) for details.

## Response Shape

Most endpoints return:

```json
{
  "ok": true,
  "data": "endpoint-specific fields"
}
```

Failures return a sanitized error shape:

```json
{
  "ok": false,
  "error": {
    "code": "runtime_error",
    "message": "Readable error message",
    "detail": "Developer detail for local debugging"
  }
}
```

## Local Development Warning

CORS is permissive for local demo convenience. Restrict `allow_origins` before adapting the backend to any network-facing environment.
