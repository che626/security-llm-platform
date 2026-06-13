import re
from collections import Counter, defaultdict
from src.simple_rag import retrieve_knowledge
from src.ai_report_generator import generate_ai_security_report


def extract_ips(text: str):
    return re.findall(r"(?:\d{1,3}\.){3}\d{1,3}", text)


def count_lines(text: str):
    return len([line for line in text.splitlines() if line.strip()])


def detect_ssh_bruteforce(lines):
    failed_by_ip = Counter()
    evidence_by_ip = defaultdict(list)

    for line in lines:
        if "Failed password" in line:
            ips = extract_ips(line)
            if ips:
                source_ip = ips[-1]
                failed_by_ip[source_ip] += 1
                evidence_by_ip[source_ip].append(line)

    alerts = []

    for ip, count in failed_by_ip.items():
        if count >= 3:
            alerts.append({
                "type": "SSH 暴力破解",
                "risk": "高危",
                "source_ip": ip,
                "evidence": evidence_by_ip[ip][:5],
                "reason": f"同一来源 IP 出现 {count} 次 SSH 登录失败"
            })

    return alerts


def detect_sql_injection(lines):
    keywords = [
        "' OR '1'='1",
        "union select",
        "select *",
        "information_schema",
        "sleep(",
        "--"
    ]

    alerts = []

    for line in lines:
        lower = line.lower()
        if any(k.lower() in lower for k in keywords):
            ips = extract_ips(line)
            alerts.append({
                "type": "SQL 注入尝试",
                "risk": "高危",
                "source_ip": ips[0] if ips else "未知",
                "evidence": [line],
                "reason": "URL 或请求内容中出现典型 SQL 注入特征"
            })

    return alerts


def detect_port_scan(lines):
    ports_by_ip = defaultdict(set)
    evidence_by_ip = defaultdict(list)

    for line in lines:
        ips = extract_ips(line)
        dpt = re.findall(r"DPT=(\d+)", line)

        if ips and dpt:
            source_ip = ips[0]
            ports_by_ip[source_ip].add(dpt[0])
            evidence_by_ip[source_ip].append(line)

    alerts = []

    for ip, ports in ports_by_ip.items():
        if len(ports) >= 4:
            alerts.append({
                "type": "端口扫描",
                "risk": "中高危",
                "source_ip": ip,
                "evidence": evidence_by_ip[ip][:6],
                "reason": f"同一来源 IP 访问了多个端口：{', '.join(sorted(ports))}"
            })

    return alerts


def analyze_log_text(log_text: str):
    lines = [line for line in log_text.splitlines() if line.strip()]

    alerts = []
    alerts.extend(detect_ssh_bruteforce(lines))
    alerts.extend(detect_sql_injection(lines))
    alerts.extend(detect_port_scan(lines))

    if not alerts:
        return "未发现明显高危攻击特征。建议继续观察日志趋势。"

    result = []
    result.append("## 日志分析结果")
    result.append(f"- 日志行数：{count_lines(log_text)}")
    result.append(f"- 命中告警数量：{len(alerts)}")
    result.append("")
    query = " ".join([alert["type"] for alert in alerts])
    knowledge = retrieve_knowledge(query)

    result.append("## RAG 知识库参考")
    result.append(knowledge)
    result.append("")

    for idx, alert in enumerate(alerts, 1):
        result.append(f"### 告警 {idx}：{alert['type']}")
        result.append(f"- 风险等级：{alert['risk']}")
        result.append(f"- 来源 IP：{alert['source_ip']}")
        result.append(f"- 判断原因：{alert['reason']}")
        result.append("- 关键证据：")

        for e in alert["evidence"]:
            result.append(f"  - {e}")

        result.append("- 建议处置：")
        result.append("  1. 临时封禁可疑来源 IP。")
        result.append("  2. 检查是否存在成功登录、异常进程或新增账户。")
        result.append("  3. 保留原始日志作为安全事件证据。")
        result.append("")

    ai_report = generate_ai_security_report(log_text, alerts)

    result.append("")
    result.append(ai_report)

    return "\n".join(result)