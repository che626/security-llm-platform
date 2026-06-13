"""
知识库管理器

功能：
1. 知识库的增删改查
2. 从多种来源导入知识
3. 知识库统计和维护
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional

from .vector_store import VectorStore

logger = logging.getLogger(__name__)


class KnowledgeManager:
    """
    知识库管理器

    管理安全知识库，支持知识的导入、检索和维护。
    """

    def __init__(
        self,
        persist_directory: str = "data/vector_db",
    ):
        """
        初始化知识库管理器

        Args:
            persist_directory: 持久化目录
        """
        self.vector_store = VectorStore(
            collection_name="security_knowledge",
            persist_directory=persist_directory,
        )

        logger.info(f"KnowledgeManager 初始化完成")
        logger.info(f"当前知识库文档数: {self.vector_store.count()}")

    def import_from_jsonl(self, filepath: str) -> int:
        """
        从 JSONL 文件导入知识

        Args:
            filepath: JSONL 文件路径

        Returns:
            导入的知识条数
        """
        if not os.path.exists(filepath):
            logger.error(f"文件不存在: {filepath}")
            return 0

        documents = []
        metadatas = []
        ids = []

        with open(filepath, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                try:
                    item = json.loads(line)

                    # 提取内容
                    instruction = item.get("instruction", "")
                    input_text = item.get("input", "")
                    output_text = item.get("output", "")

                    # 构建文档内容
                    content = f"指令：{instruction}\n"
                    if input_text:
                        content += f"输入：{input_text}\n"
                    content += f"回答：{output_text}"

                    # 提取元数据
                    attack_type = item.get("attack_type", "通用安全")
                    risk_level = item.get("risk_level", "中")

                    documents.append(content)
                    metadatas.append({
                        "source": filepath,
                        "attack_type": attack_type,
                        "risk_level": risk_level,
                        "instruction": instruction,
                    })
                    ids.append(f"knowledge_{line_num}")

                except json.JSONDecodeError as e:
                    logger.warning(f"第 {line_num} 行 JSON 解析失败: {e}")
                    continue

        # 添加到向量存储
        if documents:
            self.vector_store.add_documents(
                documents=documents,
                metadatas=metadatas,
                ids=ids,
            )
            logger.info(f"从 {filepath} 导入了 {len(documents)} 条知识")

        return len(documents)

    def import_from_markdown(self, filepath: str) -> int:
        """
        从 Markdown 文件导入知识

        Args:
            filepath: Markdown 文件路径

        Returns:
            导入的知识条数
        """
        if not os.path.exists(filepath):
            logger.error(f"文件不存在: {filepath}")
            return 0

        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()

        # 按 ## 分割章节
        sections = content.split("## ")

        documents = []
        metadatas = []
        ids = []

        for i, section in enumerate(sections, 1):
            if not section.strip():
                continue

            # 提取标题和内容
            lines = section.strip().split("\n")
            title = lines[0].strip()
            body = "\n".join(lines[1:]).strip()

            if not body:
                continue

            # 检测攻击类型
            attack_type = self._detect_attack_type(title + " " + body)
            risk_level = self._detect_risk_level(body)

            documents.append(body)
            metadatas.append({
                "source": filepath,
                "title": title,
                "attack_type": attack_type,
                "risk_level": risk_level,
            })
            ids.append(f"md_{i}")

        # 添加到向量存储
        if documents:
            self.vector_store.add_documents(
                documents=documents,
                metadatas=metadatas,
                ids=ids,
            )
            logger.info(f"从 {filepath} 导入了 {len(documents)} 条知识")

        return len(documents)

    def import_all_knowledge(self) -> int:
        """导入所有知识库文件"""
        total = 0

        # 导入 JSONL 知识库
        jsonl_files = [
            "data/knowledge/security_knowledge.jsonl",
            "data/security_instruction.jsonl",
        ]

        for filepath in jsonl_files:
            if os.path.exists(filepath):
                count = self.import_from_jsonl(filepath)
                total += count

        # 导入 Markdown 知识库
        md_files = [
            "data/knowledge_base.md",
        ]

        for filepath in md_files:
            if os.path.exists(filepath):
                count = self.import_from_markdown(filepath)
                total += count

        logger.info(f"总共导入了 {total} 条知识")
        return total

    def search(
        self,
        query: str,
        n_results: int = 5,
        filter_attack_type: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        检索知识

        Args:
            query: 查询文本
            n_results: 返回结果数量
            filter_attack_type: 过滤攻击类型

        Returns:
            检索结果列表
        """
        where = None
        if filter_attack_type:
            where = {"attack_type": filter_attack_type}

        return self.vector_store.search(
            query=query,
            n_results=n_results,
            where=where,
        )

    def get_related_knowledge(self, log_text: str, n_results: int = 3) -> str:
        """
        获取与日志相关的安全知识（用于 RAG 增强）

        Args:
            log_text: 日志文本
            n_results: 返回结果数量

        Returns:
            相关知识的格式化文本
        """
        # 检索相关知识
        results = self.search(log_text, n_results=n_results)

        if not results:
            return "未找到相关安全知识。"

        # 格式化输出
        knowledge_parts = []
        for i, result in enumerate(results, 1):
            content = result.get("content", "")
            metadata = result.get("metadata", {})
            attack_type = metadata.get("attack_type", "未知")
            risk_level = metadata.get("risk_level", "未知")

            # 截断过长的内容
            if len(content) > 500:
                content = content[:500] + "..."

            knowledge_parts.append(
                f"### 相关知识 {i}\n"
                f"- 攻击类型: {attack_type}\n"
                f"- 风险等级: {risk_level}\n"
                f"- 内容:\n{content}\n"
            )

        return "\n".join(knowledge_parts)

    def get_statistics(self) -> Dict[str, Any]:
        """获取知识库统计信息"""
        return {
            "total_knowledge": self.vector_store.count(),
            "vector_store_type": "ChromaDB",
            "persist_directory": self.vector_store.persist_directory,
        }

    def _detect_attack_type(self, text: str) -> str:
        """检测攻击类型"""
        text_lower = text.lower()

        attack_types = {
            "SSH暴力破解": ["ssh", "暴力破解", "brute force", "登录失败"],
            "SQL注入": ["sql", "注入", "injection", "union select"],
            "端口扫描": ["端口", "扫描", "port scan"],
            "C2回连": ["c2", "回连", "gate.php", "shell.php"],
            "WebShell": ["webshell", "web shell", "shell"],
            "DDoS": ["ddos", "拒绝服务", "denial of service"],
            "XSS": ["xss", "跨站脚本", "cross-site scripting"],
            "勒索软件": ["勒索", "ransomware", "加密"],
            "内部威胁": ["内部", "insider", "员工"],
            "供应链攻击": ["供应链", "supply chain", "第三方"],
        }

        for attack_type, keywords in attack_types.items():
            if any(kw in text_lower for kw in keywords):
                return attack_type

        return "通用安全"

    def _detect_risk_level(self, text: str) -> str:
        """检测风险等级"""
        text_lower = text.lower()

        if any(kw in text_lower for kw in ["高危", "high", "严重", "critical", "紧急"]):
            return "高"
        elif any(kw in text_lower for kw in ["中危", "medium", "可疑", "异常"]):
            return "中"
        elif any(kw in text_lower for kw in ["低危", "low", "信息", "建议"]):
            return "低"
        else:
            return "中"
