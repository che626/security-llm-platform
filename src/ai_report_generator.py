from src.simple_rag import retrieve_knowledge


def generate_ai_security_report(log_text, alerts):

    attack_types = [alert["type"] for alert in alerts]

    query = " ".join(attack_types)

    knowledge = retrieve_knowledge(query)

    report = []

    report.append("# AI 安全分析报告")
    report.append("")

    report.append("## 一、攻击概览")
    report.append(f"本次日志中共检测到 {len(alerts)} 个安全告警。")

    if attack_types:
        report.append(
            "涉及攻击类型：" + "、".join(set(attack_types))
        )

    report.append("")

    report.append("## 二、风险评估")

    high_risk = any(alert["risk"] == "高危" for alert in alerts)

    if high_risk:
        report.append(
            "系统判断当前事件整体风险等级为：高危。"
        )
    else:
        report.append(
            "系统判断当前事件整体风险等级为：中危。"
        )

    report.append("")
    report.append("## 三、攻击行为分析")

    for idx, alert in enumerate(alerts, 1):

        report.append(f"### {idx}. {alert['type']}")

        report.append(f"- 来源 IP：{alert['source_ip']}")
        report.append(f"- 风险等级：{alert['risk']}")
        report.append(f"- 判断原因：{alert['reason']}")

        report.append("")

    report.append("## 四、RAG 知识库参考")

    report.append(knowledge)

    report.append("")

    report.append("## 五、防御建议")

    report.append("1. 立即对可疑来源 IP 进行封禁或限制。")
    report.append("2. 检查主机是否存在成功入侵痕迹。")
    report.append("3. 检查 Web 日志、系统日志和账户行为。")
    report.append("4. 保留原始日志用于后续溯源分析。")
    report.append("5. 建议结合 IDS/IPS 和 WAF 进行持续监控。")

    report.append("")

    report.append("## 六、总结")

    report.append(
        "本次事件同时出现多种攻击行为，"
        "说明目标主机可能正在遭受自动化攻击扫描或渗透尝试。"
        "建议安全人员立即介入排查。"
    )

    return "\n".join(report)