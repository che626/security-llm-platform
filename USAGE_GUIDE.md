# Usage Guide

## Project Overview

Security LLM Platform is a local SOC AI command center demo. It combines Streamlit, FastAPI, rule-based detection, RAG security knowledge lookup, SOAR YAML generation, ATT&CK mapping, report export, and optional local model integration.

The default mode is **Rules + RAG Templates**, which runs without a model server. If Ollama or an OpenAI-compatible API is available, the assistant can call an external model and fall back safely when the model fails.

## Environment

| Item | Requirement |
|---|---|
| OS | Windows 10/11 recommended |
| Python | 3.10+ |
| Core mode | CPU is enough |
| Optional ML mode | NVIDIA GPU recommended |

## Installation

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

Optional ML stack:

```powershell
pip install -r requirements-ml.txt
```

## Running

Frontend:

```powershell
streamlit run streamlit_app.py
```

Backend:

```powershell
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```

URLs:

| Service | URL |
|---|---|
| Streamlit UI | `http://localhost:8501` |
| FastAPI docs | `http://127.0.0.1:8000/docs` |
| Health check | `http://127.0.0.1:8000/health` |

## Demo Accounts

| Username | Password | Role |
|---|---|---|
| `admin` | `Admin#2026` | Administrator |
| `analyst` | `Analyst#2026` | Security analyst |
| `researcher` | `Research#2026` | Research user |

These accounts are for local demonstration only.

## Feature Guide

### Security Dashboard

Shows total analysis count, high-risk events, unresolved incidents, risk distribution, module usage, and risk trend.

### AI Security Assistant

Answers defensive security questions with RAG knowledge references. It supports:

- SSH brute force analysis
- SQL injection triage
- port scanning interpretation
- suspicious C2 traffic reasoning
- SOAR and RAG explanation

### Log Analyzer

Accepts raw server, web, firewall, or IDS-like logs and produces:

- attack type
- risk level
- source IP
- key evidence
- RAG knowledge reference
- response recommendations

### Traffic Summary Interpreter

Explains DNS / HTTP / IDS summaries and assigns risk based on suspicious behavior such as unusual POST requests, suspicious paths, repeated connections, and legacy User-Agent strings.

### IOC Extractor

Extracts IP addresses, domains, URL paths, ports, User-Agent values, and suspicious keywords.

### Attack Chain Analysis

Groups evidence into stages such as reconnaissance, credential attack, exploitation attempt, C2 callback, and response readiness.

### ATT&CK Mapping

Maps observed behavior to ATT&CK-style tactics and techniques for cleaner incident reporting.

### SOAR Playbook Generator

Converts natural-language response requirements into YAML. Risky actions such as blocking or host isolation include manual approval.

### Report Export Center

Exports Markdown, Word-compatible `.doc`, HTML, and CSV data for local review or incident documentation.

### DeepSpeed ZeRO Experiment Console

Shows the intended research extension for memory optimization and model fine-tuning. Treat this as an experiment and demonstration module unless you have run the real training benchmarks on your hardware.

## Maintenance

Run the project check before publishing:

```powershell
python scripts/check_project.py
```

Do not commit:

- `.venv/`
- `outputs/`
- `logs/`
- local databases
- Chroma vector DB files
- model weights
- API keys or real credentials

## Security Boundary

This project is defensive only. It should not be modified to generate exploit steps, evasion guidance, credential theft workflows, or destructive automation.
