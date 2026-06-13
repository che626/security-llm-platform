from datetime import datetime


ANALYSIS_HISTORY = []


def add_history(module_name, event_type, risk_level, summary):
    record = {
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "module": module_name,
        "event_type": event_type,
        "risk_level": risk_level,
        "summary": summary
    }

    ANALYSIS_HISTORY.insert(0, record)

    if len(ANALYSIS_HISTORY) > 50:
        ANALYSIS_HISTORY.pop()


def get_history_table():
    if not ANALYSIS_HISTORY:
        return [["暂无记录", "-", "-", "-", "-"]]

    table = []

    for item in ANALYSIS_HISTORY:
        table.append([
            item["time"],
            item["module"],
            item["event_type"],
            item["risk_level"],
            item["summary"]
        ])

    return table


def get_dashboard_text():
    total = len(ANALYSIS_HISTORY)

    high_count = sum(1 for item in ANALYSIS_HISTORY if item["risk_level"] == "高危")
    middle_count = sum(1 for item in ANALYSIS_HISTORY if item["risk_level"] in ["中危", "中高危"])
    low_count = sum(1 for item in ANALYSIS_HISTORY if item["risk_level"] == "低危")

    log_count = sum(1 for item in ANALYSIS_HISTORY if item["module"] == "日志分析器")
    soar_count = sum(1 for item in ANALYSIS_HISTORY if item["module"] == "SOAR 剧本生成器")
    flow_count = sum(1 for item in ANALYSIS_HISTORY if item["module"] == "流量摘要解释器")
    chat_count = sum(1 for item in ANALYSIS_HISTORY if item["module"] == "AI 安全助手")

    return f"""
## 安全态势概览

| 指标 | 数量 |
|---|---:|
| 累计分析次数 | {total} |
| 高危事件数量 | {high_count} |
| 中危 / 中高危事件数量 | {middle_count} |
| 低危事件数量 | {low_count} |
| 日志分析次数 | {log_count} |
| SOAR 剧本生成次数 | {soar_count} |
| 流量分析次数 | {flow_count} |
| AI 助手问答次数 | {chat_count} |

### 当前态势判断

{get_risk_conclusion(high_count, middle_count, total)}
"""


def get_risk_conclusion(high_count, middle_count, total):
    if total == 0:
        return "当前暂无分析记录。可以先进入日志分析器、AI 安全助手或流量摘要解释器进行测试。"

    if high_count >= 3:
        return "当前高危事件较多，建议安全人员立即介入，优先排查暴力破解、SQL 注入、端口扫描和异常外联行为。"

    if high_count >= 1:
        return "当前存在高危事件，建议尽快确认攻击来源、影响主机和是否存在成功入侵痕迹。"

    if middle_count >= 1:
        return "当前存在中危事件，建议持续观察日志和流量变化，必要时补充防火墙或 WAF 策略。"

    return "当前未发现明显高危安全事件，系统处于相对平稳状态。"


def clear_history():
    ANALYSIS_HISTORY.clear()
    return get_history_table(), get_dashboard_text()