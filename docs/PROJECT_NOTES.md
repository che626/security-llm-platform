# Project Notes

## One-line Description

Defensive AI security analysis workbench that turns raw logs into evidence-backed findings, RAG context, SOAR playbooks, incident records and backend API outputs.

## Design Goals

- Keep the default workflow reproducible without a running model server.
- Preserve original evidence before summarizing or classifying a finding.
- Separate the Streamlit UI from FastAPI backend contracts.
- Treat RAG and model providers as explainability and enrichment layers, not as a source of unchecked authority.
- Keep SOAR execution simulated by default and mark risky actions for manual approval.

## Demo Flow

1. Log in with `admin / Admin#2026`.
2. Load demo data from the sidebar.
3. Open the dashboard and review the current risk posture.
4. Run mixed attack log analysis.
5. Review extracted IOCs and attack-chain stages.
6. Generate a SOAR playbook and check manual approval on risky actions.
7. Export a report.
8. Open backend `/docs` to inspect API surface and fallback behavior.

## Engineering Notes

- The default assistant is rule + RAG template based, not a fully fine-tuned security LLM.
- Optional model providers can be enabled through Ollama or an OpenAI-compatible API.
- DeepSpeed and LoRA modules are experiment scaffolding unless validated on a real GPU environment.
- Authentication is demo-only and must be replaced before public or production deployment.
- The large Streamlit entry file is intentionally kept stable for the current release; splitting it into smaller page modules is a planned cleanup.

## Operational Boundaries

- Use defensive logs, synthetic samples, or sanitized incident data.
- Do not connect generated SOAR actions to real infrastructure without review gates.
- Do not store real secrets, API keys, customer logs, or private infrastructure inventories in the repository.

## Suggested Repository About Section

Description:

```text
SOC-oriented AI security analysis platform with RAG, SOAR playbook generation, ATT&CK mapping, FastAPI backend, and Streamlit dashboard.
```

Topics:

```text
security soc llm rag fastapi streamlit soar mitre-attack deepspeed
```
