def explain_flow_summary(flow_text: str):
    text = flow_text.lower()

    risk_score = 0
    reasons = []

    if "post" in text:
        risk_score += 2
        reasons.append("出现 HTTP POST 请求，可能存在数据上传或回连行为。")

    if "gate.php" in text or "cmd" in text or "shell" in text:
        risk_score += 3
        reasons.append("URL 路径疑似网关、木马或控制端接口。")

    if "dns" in text and ("example" in text or "update" in text):
        risk_score += 1
        reasons.append("出现 DNS 查询和疑似更新域名，需要结合域名信誉判断。")

    if "重复" in flow_text or "20 次" in flow_text:
        risk_score += 2
        reasons.append("短时间内重复连接，可能是自动化程序或 C2 心跳。")

    if "msie 6.0" in text:
        risk_score += 1
        reasons.append("User-Agent 较老旧，可能是伪造客户端。")

    if risk_score >= 6:
        level = "高危"
    elif risk_score >= 3:
        level = "中危"
    else:
        level = "低危"

    lines = []
    lines.append("## 流量解释结果")
    lines.append(f"- 风险等级：{level}")
    lines.append(f"- 风险分数：{risk_score}")
    lines.append("- 判断依据：")

    if reasons:
        for reason in reasons:
            lines.append(f"  - {reason}")
    else:
        lines.append("  - 未发现明显异常特征。")

    lines.append("- 建议处置：")
    lines.append("  1. 查询目的 IP 和域名信誉。")
    lines.append("  2. 在防火墙或 DNS 网关中临时阻断可疑域名。")
    lines.append("  3. 检查源主机进程、计划任务和最近下载文件。")
    lines.append("  4. 保留原始流量和主机日志。")

    return "\n".join(lines)