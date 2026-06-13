# Portfolio Notes

## One-line Description

SOC-oriented AI security analysis platform with RAG, SOAR playbook generation, ATT&CK mapping, FastAPI backend, and Streamlit dashboard.

## Resume Bullets

- Built a SOC-oriented AI security analysis platform with Streamlit and FastAPI, integrating log triage, RAG security knowledge retrieval, IOC extraction, ATT&CK mapping, SOAR YAML generation, and report export.
- Designed stable offline fallback behavior using rule-based detection and local RAG templates, while supporting optional Ollama or OpenAI-compatible model providers.
- Implemented analyst workflows from detection to response: raw log analysis, attack-chain reconstruction, incident tracking, SOAR simulation, and Markdown / HTML / CSV report export.
- Added DeepSpeed ZeRO / LoRA experiment scaffolding to present a credible research extension for security model fine-tuning and memory optimization.
- Packaged the project for public GitHub release with clear documentation, setup scripts, dependency separation, release checks, and defensive security boundaries.

## Interview Story

The project started as a local security LLM demo. The packaging goal was to turn it into a credible portfolio project rather than a loose experiment. The final design focuses on three things:

1. **Product value**: a reviewer can log in, load demo data, and follow a SOC-style investigation flow.
2. **Engineering value**: FastAPI endpoints, fallback behavior, scripts, docs, and release checks make the project reproducible.
3. **Research value**: RAG and DeepSpeed/LoRA modules show a path toward model-enhanced security analysis without overstating production readiness.

## Demo Script

1. Log in with `admin / Admin#2026`.
2. Load demo data from the sidebar.
3. Show the dashboard and explain the risk posture.
4. Run mixed attack log analysis.
5. Show extracted IOCs and attack-chain stages.
6. Generate a SOAR playbook and point out manual approval on risky actions.
7. Export a report.
8. Open backend `/docs` to show API surface.

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
