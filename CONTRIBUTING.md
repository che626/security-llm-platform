# Contributing

This repository is maintained as a portfolio-grade security AI demo. Contributions are welcome when they improve clarity, reproducibility, defensive security value, or engineering quality.

## Development Setup

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

Optional ML and training dependencies are listed in `requirements-ml.txt`.

## Quality Checks

Run the release check before opening a pull request:

```bash
python scripts/check_project.py
```

## Security Boundaries

This project is defensive only. Do not add exploit instructions, evasion guidance, offensive payload generation, credential theft workflows, or real destructive automation.

Generated SOAR playbooks must keep risky actions in simulation mode or require manual approval.
