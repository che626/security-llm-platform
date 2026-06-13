# Deployment Guide

This project is designed for local portfolio review. Treat public deployment as a separate hardening project.

## Local Demo

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Optional backend:

```powershell
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```

## Optional Model Provider

Install Ollama and pull a model:

```powershell
ollama pull qwen2.5:3b
```

Then configure the app in **模型接入配置中心**.

The app remains usable without Ollama because it falls back to rule + RAG template mode.

## Publishing to GitHub

1. Create an empty public GitHub repository named `security-llm-platform`.
2. Do not initialize it with README, LICENSE, or `.gitignore`.
3. From this local project:

```powershell
git init
git branch -M main
git add .
git commit -m "Package security LLM platform portfolio project"
git remote add origin <REMOTE_URL>
git push -u origin main
```

4. Add repository topics:

```text
security, soc, llm, rag, fastapi, streamlit, soar, mitre-attack, deepspeed
```

5. Use this repository description:

```text
SOC-oriented AI security analysis platform with RAG, SOAR playbook generation, ATT&CK mapping, FastAPI backend, and Streamlit dashboard.
```

## Pre-release Checklist

Run:

```powershell
python scripts/check_project.py
```

Confirm the repository does not include:

- `.venv/`
- `outputs/`
- local DB files
- Chroma vector DB files
- generated logs
- model checkpoints
- real API keys

## Production Hardening Required

Before adapting this project to production:

- replace demo authentication with real identity management
- restrict CORS
- store secrets in a secure secret manager
- add structured audit logging
- add proper database migrations
- review every generated SOAR action
- deploy behind HTTPS and a reverse proxy
- perform security testing
