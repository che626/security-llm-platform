# Project Roadmap

This roadmap keeps the public project honest: the current repository is a local SOC AI demo, not a production security platform.

## Current Release

The current release focuses on:

- Streamlit SOC command center UI
- FastAPI backend with documented endpoints
- rule-based log and flow analysis
- RAG security knowledge lookup
- IOC extraction
- attack-chain reconstruction
- ATT&CK-style mapping
- SOAR YAML generation and simulation
- incident tracking and report export
- optional local model integration
- DeepSpeed ZeRO / LoRA research scaffolding

## Near-term Improvements

- Add screenshots under `docs/assets/`.
- Add small pytest suite for detection, IOC extraction, and SOAR generation.
- Split large Streamlit pages into focused modules.
- Add example API requests in `docs/API.md`.
- Add a lightweight Dockerfile for local operation and testing.

## Research Extensions

- Validate vector RAG with ChromaDB on a curated security knowledge set.
- Run LoRA fine-tuning on a small security instruction dataset.
- Benchmark DeepSpeed ZeRO configurations on real GPU hardware.
- Publish experiment outputs as reproducible reports, not hard-coded claims.

## Production Hardening If Reused Beyond Demo

- Replace demo login with real authentication.
- Store secrets outside the repository.
- Restrict CORS and add HTTPS.
- Use a proper database with migrations.
- Add audit logs and role-based authorization.
- Review every SOAR action before connecting to real infrastructure.
