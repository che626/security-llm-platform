# Quick Start

This guide starts the local demo in about five minutes on Windows.

## 1. Install Core Dependencies

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

Optional ML / vector / training stack:

```powershell
pip install -r requirements-ml.txt
```

## 2. Start the Frontend

```powershell
streamlit run streamlit_app.py
```

Open:

```text
http://localhost:8501
```

## 3. Optional Backend

```powershell
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```

API docs:

```text
http://127.0.0.1:8000/docs
```

## 4. Demo Accounts

| Username | Password | Role |
|---|---|---|
| `admin` | `Admin#2026` | Full demo access |
| `analyst` | `Analyst#2026` | Analysis and response workflows |
| `researcher` | `Research#2026` | RAG, evaluation, and training demos |

These are local demo credentials only.

## 5. Recommended Demo Flow

1. Log in as `admin`.
2. Click **加载示例数据** in the sidebar.
3. Open **安全态势仪表盘**.
4. Open **日志分析器** and analyze the mixed attack sample.
5. Open **IOC 威胁指标提取器** and **攻击链分析**.
6. Generate a SOAR YAML playbook and simulate it.
7. Export a report from **报告导出中心**.

## 6. Release Check

```powershell
python scripts/check_project.py
```
