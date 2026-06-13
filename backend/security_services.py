from pathlib import Path
import json
import re
from typing import List, Dict


def append_json_record(path: str, record: dict):
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    data = read_json_records(path)
    data.insert(0, record)
    data = data[:200]
    file_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def read_json_records(path: str):
    file_path = Path(path)
    if not file_path.exists():
        return []
    try:
        return json.loads(file_path.read_text(encoding="utf-8"))
    except Exception:
        return []


def extract_iocs(text: str) -> List[Dict[str, str]]:
    iocs = []

    ip_pattern = r"(?<!\d)(?:\d{1,3}\.){3}\d{1,3}(?!\d)"
    domain_pattern = r"\b(?:[a-zA-Z0-9-]+\.)+[a-zA-Z]{2,}\b"
    url_path_pattern = r"(?:GET|POST|PUT|DELETE|HEAD)\s+([^\s]+)"
    dpt_pattern = r"DPT=(\d+)"
    port_pattern = r"\bport\s+(\d{2,5})\b"
    ua_pattern = r"User-Agent[：:]\s*(.+)"
    keyword_patterns = [
        ("SQL 注入关键词", r"(?i)(union\s+select|information_schema|or\s+'?1'?\s*=\s*'?1'?|sleep\(|benchmark\()"),
        ("WebShell/C2 路径", r"(?i)(gate\.php|shell\.php|cmd=|webshell|backdoor)"),
        ("SSH 登录失败", r"Failed password"),
        ("扫描特征", r"DPT=\d+|port scan|扫描"),
    ]

    for ip in sorted(set(re.findall(ip_pattern, text))):
        iocs.append({"类型": "IP 地址", "值": ip, "说明": "从日志或流量摘要中提取到的 IP"})

    for domain in sorted(set(re.findall(domain_pattern, text))):
        if not re.match(ip_pattern, domain):
            iocs.append({"类型": "域名", "值": domain, "说明": "从 DNS、HTTP 或文本中提取到的域名"})

    for path in sorted(set(re.findall(url_path_pattern, text, flags=re.IGNORECASE))):
        iocs.append({"类型": "URL 路径", "值": path, "说明": "HTTP 请求路径"})

    ports = set(re.findall(dpt_pattern, text)) | set(re.findall(port_pattern, text, flags=re.IGNORECASE))
    for port in sorted(ports, key=lambda x: int(x)):
        iocs.append({"类型": "端口", "值": port, "说明": "日志中出现的端口"})

    for ua in re.findall(ua_pattern, text):
        iocs.append({"类型": "User-Agent", "值": ua.strip(), "说明": "可能用于识别异常客户端"})

    for name, pattern in keyword_patterns:
        for hit in sorted(set(re.findall(pattern, text))):
            value = hit if isinstance(hit, str) else " ".join(hit)
            iocs.append({"类型": name, "值": value, "说明": "命中可疑安全关键词"})

    return iocs


def build_attack_chain(text: str) -> List[Dict[str, str]]:
    lower = text.lower()
    chain = []

    def add(stage, evidence, risk, suggestion):
        chain.append({
            "阶段": stage,
            "证据": evidence,
            "风险等级": risk,
            "建议": suggestion,
        })

    if "dpt=" in lower or "port scan" in lower or "扫描" in text:
        add(
            "1. 侦察 / 端口扫描",
            "检测到多个 DPT 端口访问或扫描相关特征。",
            "中高危",
            "限制扫描源，确认暴露端口，关闭不必要服务。"
        )

    if "failed password" in lower or "登录失败" in text:
        add(
            "2. 凭证攻击 / SSH 暴力破解",
            "检测到 Failed password 或多次登录失败。",
            "高危",
            "封禁来源 IP，检查是否存在成功登录，启用密钥登录和 Fail2Ban。"
        )

    if "union select" in lower or "information_schema" in lower or "' or '1'='1" in lower:
        add(
            "3. 漏洞利用尝试 / SQL 注入",
            "检测到 OR 1=1、UNION SELECT 或 information_schema 等注入特征。",
            "高危",
            "检查 Web 日志和数据库访问，修复参数化查询，启用 WAF 规则。"
        )

    if "gate.php" in lower or "shell.php" in lower or "user-agent" in lower or "重复连接" in text:
        add(
            "4. 命令控制 / 可疑 C2 回连",
            "检测到 gate.php、shell.php、异常 User-Agent 或重复连接。",
            "高危",
            "隔离源主机，阻断可疑域名和 IP，检查进程、启动项、计划任务。"
        )

    if not chain:
        add(
            "0. 未形成明显攻击链",
            "当前输入未命中明确攻击链特征。",
            "低危",
            "建议补充更多日志、流量摘要或告警上下文。"
        )

    return chain
