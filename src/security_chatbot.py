from src.simple_rag import retrieve_knowledge


def security_chat_response(message, history):
    """
    AI 安全助手模拟回复函数。
    当前版本使用规则 + RAG 知识库生成回答。
    后续可以替换为本地大模型或 API 调用。
    """

    if not message or not message.strip():
        return "请输入你想咨询的安全问题。"

    knowledge = retrieve_knowledge(message)

    lower_msg = message.lower()

    answer = []

    answer.append("## AI 安全助手回复")
    answer.append("")

    answer.append("### 1. 问题理解")
    answer.append(f"你提出的问题是：{message}")
    answer.append("")

    if "ssh" in lower_msg or "暴力破解" in message or "登录失败" in message:
        answer.append("我判断该问题主要与 **SSH 暴力破解攻击** 有关。")
        answer.append("这类攻击通常表现为同一来源 IP 在短时间内大量尝试登录不同账户，例如 root、admin、test 等。")
    elif "sql" in lower_msg or "注入" in message:
        answer.append("我判断该问题主要与 **SQL 注入攻击** 有关。")
        answer.append("这类攻击通常通过构造恶意参数，让后端数据库执行非预期 SQL 语句。")
    elif "端口" in message or "扫描" in message:
        answer.append("我判断该问题主要与 **端口扫描行为** 有关。")
        answer.append("端口扫描通常用于探测目标主机开放的服务，是后续攻击前的重要侦察行为。")
    elif "c2" in lower_msg or "回连" in message or "流量" in message or "gate.php" in lower_msg:
        answer.append("我判断该问题可能与 **可疑 C2 回连或异常网络流量** 有关。")
        answer.append("这类行为通常表现为周期性访问陌生域名、异常 POST 请求、老旧 User-Agent 或可疑路径。")
    elif "soar" in lower_msg or "剧本" in message or "自动化" in message:
        answer.append("我判断该问题主要与 **SOAR 安全编排与自动化响应** 有关。")
        answer.append("SOAR 的核心是把告警、研判、处置和通知流程结构化，降低人工响应成本。")
    elif "rag" in lower_msg or "知识库" in message:
        answer.append("我判断该问题主要与 **RAG 检索增强生成** 有关。")
        answer.append("RAG 可以先从知识库检索相关安全知识，再辅助生成更可靠的分析结果。")
    else:
        answer.append("该问题属于一般安全分析问题，可以结合日志、流量、告警和知识库进行综合判断。")

    answer.append("")
    answer.append("### 2. 知识库检索结果")
    answer.append(knowledge)
    answer.append("")

    answer.append("### 3. 建议处理思路")
    answer.append("1. 先确认告警来源，包括日志时间、来源 IP、目标主机和攻击特征。")
    answer.append("2. 再判断是否存在真实入侵痕迹，例如成功登录、异常进程、新增账户或异常外联。")
    answer.append("3. 对高风险来源 IP、域名或 URL 进行临时阻断。")
    answer.append("4. 保留原始日志、流量和主机证据，方便后续溯源。")
    answer.append("5. 如果涉及自动化处置，应先生成 SOAR 剧本，并对封禁、隔离等高风险动作设置人工确认。")
    answer.append("")

    answer.append("### 4. 系统说明")
    answer.append("当前助手使用的是 **规则判断 + RAG 知识库检索 + 模板化 AI 报告生成**。")
    answer.append("后续可以把这一层替换为 Qwen、DeepSeek、ChatGLM 或其他本地大模型。")

    return "\n".join(answer)