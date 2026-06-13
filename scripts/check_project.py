from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

PYTHON_FILES = [
    "streamlit_app.py",
    "frontend_api_client.py",
    "backend/main.py",
    "backend/model_client.py",
    "backend/security_services.py",
    "src/log_analyzer.py",
    "src/soar_generator.py",
    "src/flow_explainer.py",
    "src/security_chatbot.py",
    "src/simple_rag.py",
]

FORBIDDEN_PATHS = [
    ".venv",
    "__pycache__",
    ".superpowers",
    "outputs",
    "logs",
    "data/vector_db",
]

SECRET_PATTERNS = [
    re.compile(r"sk-[A-Za-z0-9_\-]{20,}"),
    re.compile(r"api[_-]?key\s*=\s*['\"][^'\"]{12,}['\"]", re.IGNORECASE),
    re.compile(r"bearer\s+[A-Za-z0-9_\-.]{20,}", re.IGNORECASE),
]

ALLOWLISTED_TEXT = {
    "Admin#2026",
    "Analyst#2026",
    "Research#2026",
    "your-secret-key-change-in-production",
    "change-me-for-local-development",
}


def run(cmd: list[str]) -> tuple[int, str]:
    proc = subprocess.run(
        cmd,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    return proc.returncode, proc.stdout


def check_compile() -> list[str]:
    existing = [str(ROOT / f) for f in PYTHON_FILES if (ROOT / f).exists()]
    code, output = run([sys.executable, "-m", "py_compile", *existing])
    return [] if code == 0 else [f"Python compile failed:\n{output}"]


def check_smoke() -> list[str]:
    script = """
from src.log_analyzer import analyze_log_text
from src.soar_generator import generate_soar_yaml, simulate_playbook
from src.flow_explainer import explain_flow_summary
from src.simple_rag import retrieve_knowledge

log = "May 11 10:01:02 server sshd[1001]: Failed password for root from 8.8.8.8 port 53422 ssh2\\n" * 3
assert "SSH" in analyze_log_text(log)
yaml_text = generate_soar_yaml("检测到 SSH 暴力破解后通知管理员并封禁来源 IP")
assert "manual_approval" in yaml_text
assert "模拟运行" in simulate_playbook(yaml_text)
assert "风险等级" in explain_flow_summary("HTTP POST /gate.php User-Agent: MSIE 6.0 重复连接 20 次")
assert retrieve_knowledge("SSH 暴力破解")
"""
    code, output = run([sys.executable, "-c", script])
    return [] if code == 0 else [f"Smoke tests failed:\n{output}"]


def check_forbidden_paths() -> list[str]:
    issues = []
    for rel in FORBIDDEN_PATHS:
        path = ROOT / rel
        if path.exists():
            issues.append(f"Generated/local path exists and should not be committed: {rel}")
    return issues


def check_secret_patterns() -> list[str]:
    issues = []
    searchable_exts = {".py", ".md", ".txt", ".json", ".yml", ".yaml", ".bat", ".example"}
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        if any(part in {".venv", "__pycache__", ".git", ".superpowers"} for part in path.parts):
            continue
        if path.suffix.lower() not in searchable_exts and path.name != ".env.example":
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for pattern in SECRET_PATTERNS:
            for match in pattern.findall(text):
                if match in ALLOWLISTED_TEXT:
                    continue
                issues.append(f"Potential secret in {path.relative_to(ROOT)}: {match[:24]}...")
    return issues


def main() -> int:
    checks = {
        "compile": check_compile(),
        "smoke": check_smoke(),
        "generated_paths": check_forbidden_paths(),
        "secret_scan": check_secret_patterns(),
    }

    print(json.dumps(checks, ensure_ascii=False, indent=2))

    blocking = checks["compile"] + checks["smoke"] + checks["secret_scan"]
    if blocking:
        return 1

    if checks["generated_paths"]:
        print("\nNote: generated paths are ignored by .gitignore, but clean them before creating a polished release archive.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
