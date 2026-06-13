"""
安全指令数据集加载器

支持从多种格式加载安全指令数据：
1. JSONL 格式（instruction/input/output）
2. CSV 格式
3. HuggingFace 数据集
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class SecurityDatasetLoader:
    """安全数据集加载器"""

    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        logger.info(f"SecurityDatasetLoader 初始化，数据目录: {data_dir}")

    def load_jsonl(self, filename: str) -> List[Dict[str, Any]]:
        """加载 JSONL 格式数据"""
        filepath = os.path.join(self.data_dir, filename)

        if not os.path.exists(filepath):
            logger.warning(f"文件不存在: {filepath}")
            return []

        data = []
        with open(filepath, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                    data.append(item)
                except json.JSONDecodeError as e:
                    logger.warning(f"第 {line_num} 行 JSON 解析失败: {e}")

        logger.info(f"从 {filename} 加载了 {len(data)} 条数据")
        return data

    def save_jsonl(self, data: List[Dict[str, Any]], filename: str):
        """保存 JSONL 格式数据"""
        filepath = os.path.join(self.data_dir, filename)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        with open(filepath, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        logger.info(f"保存了 {len(data)} 条数据到 {filename}")

    def generate_security_instruction_dataset(self) -> List[Dict[str, Any]]:
        """生成安全指令数据集"""
        dataset = [
            # SSH 暴力破解相关
            {
                "instruction": "请分析以下 SSH 日志中的安全威胁",
                "input": "May 11 10:01:02 server sshd[1001]: Failed password for root from 8.8.8.8 port 53422 ssh2",
                "output": "该日志显示 SSH 暴力破解攻击特征：\n\n1. 攻击特征：同一来源 IP 短时间内多次登录失败\n2. 目标用户：root（高权限账户）\n3. 来源 IP：8.8.8.8（外部 IP）\n\n处置建议：\n1. 临时封禁来源 IP\n2. 检查是否存在成功登录记录\n3. 启用 Fail2Ban 防暴力破解\n4. 禁止 root 远程登录\n5. 使用密钥认证替代密码认证"
            },
            {
                "instruction": "请解释 SSH 暴力破解攻击的原理和防御方法",
                "input": "",
                "output": "SSH 暴力破解攻击原理：\n\n攻击者通过自动化工具尝试大量用户名和密码组合，试图猜测正确的 SSH 登录凭据。常见特征包括：\n1. 短时间内大量登录失败\n2. 尝试常见用户名（root, admin, test）\n3. 来自同一 IP 的持续尝试\n\n防御方法：\n1. 启用 Fail2Ban 自动封禁\n2. 使用密钥认证\n3. 禁止 root 远程登录\n4. 修改默认 SSH 端口\n5. 限制登录尝试次数\n6. 使用双因素认证"
            },

            # SQL 注入相关
            {
                "instruction": "请分析以下 Web 日志中的 SQL 注入攻击",
                "input": "1.2.3.4 - - \"GET /login.php?id=1' OR '1'='1 HTTP/1.1\" 200",
                "output": "该日志显示 SQL 注入攻击尝试：\n\n1. 攻击特征：URL 参数中包含 ' OR '1'='1\n2. 攻击类型：布尔盲注\n3. 目标：登录页面 login.php\n4. 来源 IP：1.2.3.4\n\n处置建议：\n1. 使用参数化查询\n2. 启用 WAF 规则\n3. 审计数据库访问日志\n4. 检查是否有数据泄露\n5. 更新 Web 应用防火墙规则"
            },
            {
                "instruction": "请解释 SQL 注入攻击的原理和防御方法",
                "input": "",
                "output": "SQL 注入攻击原理：\n\n攻击者通过在输入字段中插入恶意 SQL 代码，利用应用程序对用户输入缺乏验证的漏洞，执行非预期的数据库操作。\n\n常见类型：\n1. 联合查询注入（UNION SELECT）\n2. 布尔盲注（AND 1=1）\n3. 时间盲注（SLEEP()）\n4. 报错注入\n\n防御方法：\n1. 使用参数化查询/预编译语句\n2. 输入验证和过滤\n3. 最小权限原则\n4. 启用 WAF\n5. 定期安全审计"
            },

            # 端口扫描相关
            {
                "instruction": "请分析以下防火墙日志中的端口扫描行为",
                "input": "SRC=5.5.5.5 DST=192.168.1.10 DPT=22 ACTION=DENY\nSRC=5.5.5.5 DST=192.168.1.10 DPT=80 ACTION=DENY\nSRC=5.5.5.5 DST=192.168.1.10 DPT=443 ACTION=DENY\nSRC=5.5.5.5 DST=192.168.1.10 DPT=3306 ACTION=DENY",
                "output": "该日志显示端口扫描行为：\n\n1. 扫描特征：同一来源 IP 访问多个不同端口\n2. 扫描端口：22(SSH)、80(HTTP)、443(HTTPS)、3306(MySQL)\n3. 来源 IP：5.5.5.5\n4. 目标 IP：192.168.1.10\n\n处置建议：\n1. 封禁扫描源 IP\n2. 检查暴露的服务\n3. 关闭不必要的端口\n4. 增加 IDS/IPS 规则\n5. 配置防火墙限制访问"
            },

            # C2 回连相关
            {
                "instruction": "请分析以下网络流量中的可疑 C2 回连行为",
                "input": "时间：2026-05-11 10:25:01\n源 IP：192.168.1.23\n目的 IP：45.9.148.10\n协议：DNS + HTTP\nDNS 查询：abc-update-check.example\nHTTP 请求：POST /gate.php HTTP/1.1\nUser-Agent：Mozilla/4.0 (compatible; MSIE 6.0)\n请求频率：1 分钟内重复连接 20 次",
                "output": "该流量显示可疑 C2 回连特征：\n\n1. 可疑特征：\n   - POST 请求到 gate.php（常见 C2 路径）\n   - 异常 User-Agent（MSIE 6.0 过时浏览器）\n   - 高频率重复连接（1 分钟 20 次）\n   - 陌生域名（abc-update-check.example）\n\n2. 风险等级：高危\n\n处置建议：\n1. 立即隔离源主机（192.168.1.23）\n2. 阻断目的域名和 IP\n3. 检查主机进程和计划任务\n4. 检查启动项和最近下载文件\n5. 保留流量证据用于溯源"
            },

            # SOAR 剧本相关
            {
                "instruction": "请为 SSH 暴力破解事件生成 SOAR 响应剧本",
                "input": "检测到同一 IP 在 5 分钟内 SSH 登录失败超过 10 次",
                "output": "SOAR 响应剧本：\n\n触发条件：同一 IP 5 分钟内 SSH 失败 >= 10 次\n\n响应动作：\n1. 自动封禁来源 IP 24 小时\n2. 通知安全管理员\n3. 收集主机认证日志\n4. 检查是否存在成功登录\n5. 检查是否有关联的异常进程\n\n人工确认：\n- 封禁操作需要人工确认\n- 确认后自动执行\n\n后续动作：\n- 记录事件到处置中心\n- 更新威胁情报库\n- 生成处置报告"
            },
            {
                "instruction": "请为 SQL 注入攻击生成 SOAR 响应剧本",
                "input": "Web 日志检测到 SQL 注入攻击特征",
                "output": "SOAR 响应剧本：\n\n触发条件：Web 日志中检测到 SQL 注入特征\n\n响应动作：\n1. 阻断攻击源 IP\n2. 启用 WAF 严格模式\n3. 通知 Web 应用管理员\n4. 收集 Web 访问日志\n5. 审计数据库查询日志\n\n人工确认：\n- WAF 规则更新需要人工确认\n- 数据库审计需要人工确认范围\n\n后续动作：\n- 检查是否有数据泄露\n- 更新 WAF 规则库\n- 代码安全审计\n- 生成安全报告"
            },

            # 攻击链分析相关
            {
                "instruction": "请分析以下日志中的攻击链",
                "input": "日志包含：端口扫描（5.5.5.5 访问多个端口）、SSH 暴力破解（8.8.8.8 多次登录失败）、SQL 注入（1.2.3.4 注入尝试）、可疑 C2 回连（192.168.1.23 访问 gate.php）",
                "output": "攻击链分析：\n\n阶段 1：侦察/端口扫描\n- 证据：5.5.5.5 访问 22、80、443、3306 端口\n- 风险：中高危\n\n阶段 2：凭证攻击/SSH 暴力破解\n- 证据：8.8.8.8 多次 SSH 登录失败\n- 风险：高危\n\n阶段 3：漏洞利用/SQL 注入\n- 证据：1.2.3.4 注入 ' OR '1'='1\n- 风险：高危\n\n阶段 4：命令控制/C2 回连\n- 证据：192.168.1.23 POST gate.php\n- 风险：高危\n\n综合风险：高危\n建议：立即进入事件处置流程"
            },

            # RAG 知识库相关
            {
                "instruction": "请解释 RAG 知识库在安全分析系统中的作用",
                "input": "",
                "output": "RAG（检索增强生成）在安全分析系统中的作用：\n\n1. 知识检索\n   - 检索相关安全知识和处置建议\n   - 提供攻击特征和检测方法\n   - 关联历史事件和响应经验\n\n2. 增强分析\n   - 结合知识库提升分析准确性\n   - 减少模型幻觉\n   - 提供可追溯的分析依据\n\n3. 应用场景\n   - 日志分析：检索攻击特征库\n   - 事件响应：检索处置剧本\n   - 威胁情报：检索 IOC 信息\n   - 合规检查：检索安全标准\n\n4. 技术实现\n   - 向量数据库存储知识\n   - 嵌入模型进行语义检索\n   - 上下文注入提升回答质量"
            },
        ]

        return dataset

    def generate_attack_pattern_dataset(self) -> List[Dict[str, Any]]:
        """生成攻击模式数据集"""
        return [
            {
                "attack_type": "SSH 暴力破解",
                "patterns": ["Failed password", "Invalid user", "authentication failure"],
                "risk_level": "high",
                "mitre_attack_id": "T1110",
                "mitre_attack_name": "Brute Force",
                "detection_rules": [
                    "同一 IP 5 分钟内登录失败 >= 5 次",
                    "同一用户名 10 分钟内登录失败 >= 10 次"
                ],
                "response_actions": [
                    "封禁来源 IP",
                    "检查成功登录记录",
                    "启用 Fail2Ban",
                    "通知安全管理员"
                ]
            },
            {
                "attack_type": "SQL 注入",
                "patterns": ["' OR '1'='1", "UNION SELECT", "information_schema", "sleep(", "benchmark("],
                "risk_level": "high",
                "mitre_attack_id": "T1190",
                "mitre_attack_name": "Exploit Public-Facing Application",
                "detection_rules": [
                    "URL 参数中包含 SQL 关键字",
                    "POST 数据中包含 SQL 注入特征",
                    "数据库错误信息泄露"
                ],
                "response_actions": [
                    "阻断攻击请求",
                    "启用 WAF 规则",
                    "审计数据库日志",
                    "代码安全审查"
                ]
            },
            {
                "attack_type": "端口扫描",
                "patterns": ["DPT=", "SYN", "multiple ports", "sequential scan"],
                "risk_level": "medium",
                "mitre_attack_id": "T1046",
                "mitre_attack_name": "Network Service Discovery",
                "detection_rules": [
                    "同一来源访问 >= 3 个不同端口",
                    "短时间内大量 SYN 包",
                    "顺序端口扫描模式"
                ],
                "response_actions": [
                    "封禁扫描源",
                    "检查暴露服务",
                    "配置防火墙规则",
                    "增加 IDS 规则"
                ]
            },
            {
                "attack_type": "C2 回连",
                "patterns": ["gate.php", "shell.php", "POST", "MSIE 6.0", "periodic connection"],
                "risk_level": "critical",
                "mitre_attack_id": "T1071",
                "mitre_attack_name": "Application Layer Protocol",
                "detection_rules": [
                    "访问已知 C2 路径",
                    "异常 User-Agent",
                    "周期性连接模式",
                    "高频率重复请求"
                ],
                "response_actions": [
                    "隔离受感染主机",
                    "阻断 C2 域名/IP",
                    "检查恶意进程",
                    "收集取证证据"
                ]
            },
        ]

    def build_training_data(self) -> List[Dict[str, Any]]:
        """构建训练数据"""
        # 加载安全指令数据
        instruction_data = self.generate_security_instruction_dataset()

        # 加载攻击模式数据
        attack_patterns = self.generate_attack_pattern_dataset()

        # 转换为训练格式
        training_data = []

        # 指令数据直接使用
        training_data.extend(instruction_data)

        # 攻击模式数据转换为问答格式
        for pattern in attack_patterns:
            training_data.append({
                "instruction": f"请分析 {pattern['attack_type']} 攻击的特征和防御方法",
                "input": f"攻击模式：{', '.join(pattern['patterns'])}",
                "output": f"攻击类型：{pattern['attack_type']}\n风险等级：{pattern['risk_level']}\nMITRE ATT&CK：{pattern['mitre_attack_id']} - {pattern['mitre_attack_name']}\n\n检测规则：\n" + "\n".join(f"- {r}" for r in pattern['detection_rules']) + "\n\n响应动作：\n" + "\n".join(f"- {a}" for a in pattern['response_actions'])
            })

        return training_data

    def prepare_dataset(self, output_filename: str = "security_instruction.jsonl"):
        """准备数据集"""
        # 构建训练数据
        training_data = self.build_training_data()

        # 保存到文件
        self.save_jsonl(training_data, output_filename)

        logger.info(f"数据集准备完成，共 {len(training_data)} 条数据")
        return training_data


if __name__ == "__main__":
    loader = SecurityDatasetLoader()
    loader.prepare_dataset()
