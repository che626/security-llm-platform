from fastapi.testclient import TestClient

from backend.main import app
from backend.security_services import build_attack_chain, extract_iocs
from src.log_analyzer import analyze_log_text
from src.security_chatbot import security_chat_response
from src.soar_generator import generate_soar_yaml, simulate_playbook


SAMPLE_LOG = """May 11 10:01:02 server sshd[1001]: Failed password for root from 8.8.8.8 port 53422 ssh2
May 11 10:01:06 server sshd[1002]: Failed password for admin from 8.8.8.8 port 53423 ssh2
May 11 10:01:10 server sshd[1003]: Failed password for test from 8.8.8.8 port 53424 ssh2
May 11 10:02:22 web nginx: 1.2.3.4 - - "GET /login.php?id=1' OR '1'='1 HTTP/1.1" 200
"""


def test_log_analysis_returns_evidence_and_risk():
    report = analyze_log_text(SAMPLE_LOG)

    assert "日志分析结果" in report
    assert "SSH 暴力破解" in report
    assert "SQL 注入" in report
    assert "高危" in report


def test_ioc_extraction_and_attack_chain():
    iocs = extract_iocs(SAMPLE_LOG)
    chain = build_attack_chain(SAMPLE_LOG)

    assert any(item["类型"] == "IP 地址" and item["值"] == "8.8.8.8" for item in iocs)
    assert any("凭证攻击" in item["阶段"] for item in chain)
    assert any("SQL 注入" in item["阶段"] for item in chain)


def test_soar_generation_keeps_risky_actions_manual():
    yaml_text = generate_soar_yaml("检测到 SSH 暴力破解后通知管理员并封禁来源 IP 24 小时")
    simulation = simulate_playbook(yaml_text)

    assert "block_source_ip" in yaml_text
    assert "manual_approval: true" in yaml_text
    assert "不会真实执行任何命令" in simulation


def test_chatbot_uses_rule_rag_fallback():
    answer = security_chat_response("SSH 暴力破解应该怎么处理？", [])

    assert "规则判断" in answer
    assert "知识库检索结果" in answer
    assert "SSH" in answer


def test_fastapi_health_and_core_endpoints():
    client = TestClient(app)

    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["ok"] is True

    analysis = client.post("/api/log/analyze", json={"text": SAMPLE_LOG})
    assert analysis.status_code == 200
    body = analysis.json()
    assert body["ok"] is True
    assert "report" in body
    assert body["iocs"]

    soar = client.post("/api/soar/generate", json={"requirement": "检测到 SSH 暴力破解后通知管理员"})
    assert soar.status_code == 200
    assert soar.json()["ok"] is True
