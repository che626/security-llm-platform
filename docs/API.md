# API Reference

Start the backend:

```powershell
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```

Interactive docs:

```text
http://127.0.0.1:8000/docs
```

## Health

### `GET /health`

Returns service status.

```json
{
  "ok": true,
  "service": "security-llm-backend",
  "time": "2026-06-13 14:00:00"
}
```

## Model Configuration

### `GET /api/model/config`

Reads the local model provider configuration.

### `POST /api/model/config`

```json
{
  "mode": "ollama",
  "ollama_url": "http://localhost:11434",
  "model_name": "qwen2.5:3b"
}
```

### `POST /api/model/test`

Tests connectivity to the configured Ollama model.

## Assistant

### `POST /api/chat`

```json
{
  "message": "SSH 登录失败很多次应该怎么处理？",
  "history": [],
  "use_model": true
}
```

Returns:

- response mode
- RAG knowledge text
- assistant answer
- sanitized fallback error when applicable

## Security Analysis

### `POST /api/log/analyze`

```json
{
  "text": "May 11 10:01:02 server sshd[1001]: Failed password for root from 8.8.8.8 port 53422 ssh2"
}
```

Returns:

- Markdown report
- IOC list
- attack-chain list

### `POST /api/flow/explain`

Explains DNS / HTTP / IDS-style traffic summaries.

### `POST /api/ioc/extract`

Extracts IPs, domains, URL paths, ports, User-Agent values, and suspicious keywords.

### `POST /api/attack-chain/analyze`

Builds a lightweight attack-chain summary from logs or alerts.

## RAG

### `POST /api/rag/retrieve`

```json
{
  "query": "SQL 注入处置建议"
}
```

Returns matching local knowledge snippets.

## SOAR

### `POST /api/soar/generate`

```json
{
  "requirement": "检测到 SSH 暴力破解后通知管理员并封禁来源 IP 24 小时"
}
```

Returns a YAML playbook.

### `POST /api/soar/simulate`

```json
{
  "yaml_text": "playbook_name: ssh_response\n..."
}
```

Returns a simulated execution plan. No real blocking, isolation, or host commands are executed.

## Local Development Notes

The backend currently enables permissive CORS for local development. Restrict `allow_origins` before any internet-facing deployment.
