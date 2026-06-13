import yaml
import re


def normalize_name(text: str):
    name = re.sub(r"[^a-zA-Z0-9_]+", "_", text.lower())
    name = name.strip("_")
    return name[:50] or "security_response_playbook"


def generate_soar_yaml(requirement: str):
    req = requirement.lower()

    playbook = {
        "playbook_name": normalize_name(requirement),
        "description": requirement,
        "trigger": {
            "type": "security_alert",
            "conditions": []
        },
        "actions": [],
        "safety": {
            "manual_approval_required": True,
            "note": "所有高风险动作只做模拟运行，真实执行前必须人工确认。"
        }
    }

    if "ssh" in req or "暴力破解" in requirement or "登录失败" in requirement:
        playbook["trigger"]["conditions"].append({
            "alert_type": "ssh_failed_login",
            "threshold": ">= 5",
            "window": "5m"
        })

    if "sql" in req or "注入" in requirement:
        playbook["trigger"]["conditions"].append({
            "alert_type": "sql_injection",
            "threshold": ">= 1",
            "window": "10m"
        })

    if "端口扫描" in requirement or "扫描" in requirement:
        playbook["trigger"]["conditions"].append({
            "alert_type": "port_scan",
            "threshold": ">= 4 ports",
            "window": "5m"
        })

    if "cpu" in req or "高负载" in requirement:
        playbook["trigger"]["conditions"].append({
            "metric": "cpu_usage",
            "operator": ">",
            "value": 90,
            "duration": "5m"
        })

    if "通知" in requirement or "告警" in requirement:
        playbook["actions"].append({
            "name": "notify_security_team",
            "type": "notification",
            "channel": "email",
            "message": "检测到安全告警，请安全人员确认。"
        })

    if "封禁" in requirement or "拉黑" in requirement or "阻断" in requirement:
        playbook["actions"].append({
            "name": "block_source_ip",
            "type": "firewall_block",
            "target": "{{ source_ip }}",
            "duration": "24h",
            "manual_approval": True
        })

    if "隔离" in requirement:
        playbook["actions"].append({
            "name": "isolate_host",
            "type": "host_isolation",
            "target": "{{ affected_host }}",
            "manual_approval": True
        })

    playbook["actions"].append({
        "name": "collect_evidence",
        "type": "evidence_collection",
        "commands": [
            "who",
            "last -n 20",
            "ps aux --sort=-%cpu | head",
            "journalctl --since '30 minutes ago'"
        ],
        "manual_approval": False
    })

    return yaml.safe_dump(playbook, allow_unicode=True, sort_keys=False)


def validate_yaml(yaml_text: str):
    try:
        yaml.safe_load(yaml_text)
        return True, "YAML 格式合法"
    except Exception as e:
        return False, f"YAML 格式错误：{e}"


def simulate_playbook(yaml_text: str):
    ok, msg = validate_yaml(yaml_text)

    if not ok:
        return msg

    data = yaml.safe_load(yaml_text)

    lines = [
        "## 模拟运行结果",
        "不会真实执行任何命令，只展示计划动作。",
        ""
    ]

    for i, action in enumerate(data.get("actions", []), 1):
        lines.append(f"{i}. 动作：{action.get('name')}")
        lines.append(f"   类型：{action.get('type')}")
        lines.append(f"   是否需要人工确认：{action.get('manual_approval', False)}")
        lines.append("")

    return "\n".join(lines)