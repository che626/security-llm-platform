# Portfolio Notes

## One-line Description

Defensive AI security analysis workbench that turns raw logs into evidence-backed findings, RAG context, SOAR playbooks, incident records and backend API outputs.

## Resume Bullets

- Built a SOC-style AI security workbench with Streamlit and FastAPI, covering log triage, IOC extraction, RAG knowledge lookup, ATT&CK-style mapping, SOAR YAML generation and report export.
- Designed an offline-safe rule + RAG fallback path so the assistant remains reproducible without a running model server, while still supporting optional Ollama / OpenAI-compatible providers.
- Exposed the same analysis workflow through documented FastAPI endpoints and added pytest + CI checks for core security services and API behavior.
- Modeled safe automation boundaries by keeping SOAR execution simulated and requiring manual approval for risky response actions.
- Documented DeepSpeed ZeRO / LoRA as a research extension rather than overstating unvalidated model training claims.

## Interview Story

The project started as a local security LLM demo. I reframed it as an analyst workbench because a hiring team needs to see more than a chatbot: they need evidence handling, backend interfaces, failure modes, and security boundaries. The final design focuses on three things:

1. **Product judgment**: a reviewer can follow a realistic triage flow instead of clicking disconnected features.
2. **Backend credibility**: API endpoints, fallback behavior, tests, scripts and docs make the project inspectable.
3. **Security judgment**: the system is defensive, approval-aware, and honest about demo vs production boundaries.

## Demo Script

1. Log in with `admin / Admin#2026`.
2. Load demo data from the sidebar.
3. Show the dashboard and explain the risk posture.
4. Run mixed attack log analysis.
5. Show extracted IOCs and attack-chain stages.
6. Generate a SOAR playbook and point out manual approval on risky actions.
7. Export a report.
8. Open backend `/docs` to show API surface and explain the fallback contract.

## Strong Interview Angles

- Why rule + RAG fallback is useful for demos and incident tooling.
- How to avoid overstating LLM capability when the default system is template-backed.
- How SOAR generation can be useful while still preventing unsafe automation.
- Where FastAPI helps separate UI experimentation from backend contracts.
- What would need to change before production: authentication, CORS, audit logs, secrets, persistence and real action review.

## Likely Follow-up Questions

| Question | Good answer direction |
|---|---|
| Is this a real fine-tuned LLM? | No. The default is rule + RAG fallback; model providers are optional. DeepSpeed/LoRA is documented as an extension path. |
| What makes it safe? | Defensive scope, simulated SOAR execution, manual approval flags, no exploit generation, documented demo credentials. |
| Why Streamlit? | Fast iteration for a portfolio workbench; backend contracts remain in FastAPI so the core logic can move later. |
| What would you improve next? | Split large Streamlit file, add persistent DB migrations, strengthen tests, validate vector RAG and run real GPU benchmarks. |

## Honest Limitations

- The default assistant is rule + RAG template based, not a fully fine-tuned security LLM.
- DeepSpeed and LoRA modules are experiment scaffolding unless validated on a real GPU environment.
- Authentication is demo-only.
- SOAR actions are simulated and should not be wired to production systems without review.

## Suggested GitHub About Section

Description:

```text
SOC-oriented AI security analysis platform with RAG, SOAR playbook generation, ATT&CK mapping, FastAPI backend, and Streamlit dashboard.
```

Topics:

```text
security soc llm rag fastapi streamlit soar mitre-attack deepspeed
```
