"""
安全知识检索器

针对安全场景优化的检索器，支持：
1. 安全术语识别
2. 攻击类型匹配
3. 多维度检索
4. 结果重排序
"""

import re
import logging
from typing import List, Dict, Any, Optional, Tuple
from .vector_store import VectorStore

logger = logging.getLogger(__name__)


class SecurityRetriever:
    """
    安全知识检索器

    针对安全场景优化的检索器，能够识别安全术语和攻击类型，
    并从知识库中检索最相关的安全知识。
    """

    # 安全术语词典
    SECURITY_TERMS = {
        # 攻击类型
        "ssh暴力破解": ["SSH", "暴力破解", "brute force", "登录失败"],
        "sql注入": ["SQL", "注入", "injection", "UNION SELECT"],
        "端口扫描": ["端口", "扫描", "port scan", "nmap"],
        "c2回连": ["C2", "回连", "command and control", "gate.php"],
        "webshell": ["WebShell", "web shell", "shell.php"],
        "ddos": ["DDoS", "分布式拒绝服务", "denial of service"],
        "xss": ["XSS", "跨站脚本", "cross-site scripting"],
        "csrf": ["CSRF", "跨站请求伪造", "cross-site request forgery"],

        # 安全工具
        "fail2ban": ["Fail2Ban", "fail2ban", "自动封禁"],
        "waf": ["WAF", "web application firewall", "web应用防火墙"],
        "ids": ["IDS", "入侵检测系统", "intrusion detection"],
        "ips": ["IPS", "入侵防御系统", "intrusion prevention"],
        "siem": ["SIEM", "安全信息和事件管理"],

        # 安全概念
        "ioc": ["IOC", "威胁指标", "indicator of compromise"],
        "mitre": ["MITRE", "ATT&CK", "攻击矩阵"],
        "soar": ["SOAR", "安全编排自动化响应"],
        "rag": ["RAG", "检索增强生成", "retrieval augmented generation"],
        "zero_day": ["零日漏洞", "zero-day", "0day"],
    }

    # 攻击类型关键词映射
    ATTACK_KEYWORDS = {
        "ssh暴力破解": ["Failed password", "Invalid user", "authentication failure", "SSH"],
        "sql注入": ["' OR '1'='1", "UNION SELECT", "information_schema", "sleep(", "SQL"],
        "端口扫描": ["DPT=", "port", "scan", "SYN"],
        "c2回连": ["gate.php", "shell.php", "POST", "C2", "MSIE 6.0"],
    }

    def __init__(self, vector_store: VectorStore):
        """
        初始化检索器

        Args:
            vector_store: 向量存储实例
        """
        self.vector_store = vector_store
        logger.info("SecurityRetriever 初始化完成")

    def _extract_security_terms(self, query: str) -> List[str]:
        """从查询中提取安全术语"""
        terms = []
        query_lower = query.lower()

        for term, aliases in self.SECURITY_TERMS.items():
            for alias in aliases:
                if alias.lower() in query_lower:
                    terms.append(term)
                    break

        return terms

    def _detect_attack_type(self, query: str) -> List[str]:
        """检测查询中涉及的攻击类型"""
        attack_types = []
        query_lower = query.lower()

        for attack_type, keywords in self.ATTACK_KEYWORDS.items():
            for keyword in keywords:
                if keyword.lower() in query_lower:
                    attack_types.append(attack_type)
                    break

        return attack_types

    def _expand_query(self, query: str) -> str:
        """扩展查询，添加相关术语"""
        terms = self._extract_security_terms(query)
        attack_types = self._detect_attack_type(query)

        # 构建扩展查询
        expanded_parts = [query]

        # 添加安全术语
        for term in terms:
            if term not in query.lower():
                expanded_parts.append(term)

        # 添加攻击类型相关术语
        for attack_type in attack_types:
            aliases = self.SECURITY_TERMS.get(attack_type, [])
            for alias in aliases[:2]:  # 只取前两个别名
                if alias.lower() not in query.lower():
                    expanded_parts.append(alias)

        return " ".join(expanded_parts)

    def retrieve(
        self,
        query: str,
        n_results: int = 5,
        filter_attack_type: Optional[str] = None,
        filter_risk_level: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        检索安全知识

        Args:
            query: 查询文本
            n_results: 返回结果数量
            filter_attack_type: 过滤攻击类型
            filter_risk_level: 过滤风险等级

        Returns:
            检索结果列表
        """
        # 扩展查询
        expanded_query = self._expand_query(query)
        logger.info(f"原始查询: {query}")
        logger.info(f"扩展查询: {expanded_query}")

        # 构建过滤条件
        where = {}
        if filter_attack_type:
            where["attack_type"] = filter_attack_type
        if filter_risk_level:
            where["risk_level"] = filter_risk_level

        # 检索
        results = self.vector_store.search(
            query=expanded_query,
            n_results=n_results,
            where=where if where else None,
        )

        # 重排序
        results = self._rerank_results(query, results)

        return results

    def _rerank_results(
        self,
        query: str,
        results: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """重排序结果"""
        if not results:
            return results

        query_lower = query.lower()
        query_terms = set(query_lower.split())

        scored_results = []
        for result in results:
            content = result.get("content", "").lower()
            metadata = result.get("metadata", {})

            # 基础分数（来自向量检索）
            base_score = 1 - result.get("distance", 0)

            # 关键词匹配加分
            content_terms = set(content.split())
            keyword_overlap = len(query_terms.intersection(content_terms))
            keyword_score = keyword_overlap / max(len(query_terms), 1)

            # 安全术语匹配加分
            security_terms = self._extract_security_terms(query)
            term_score = 0
            for term in security_terms:
                if term.lower() in content:
                    term_score += 0.1

            # 攻击类型匹配加分
            attack_types = self._detect_attack_type(query)
            attack_score = 0
            for attack_type in attack_types:
                if attack_type.lower() in content:
                    attack_score += 0.2

            # 计算最终分数
            final_score = (
                base_score * 0.5 +
                keyword_score * 0.2 +
                term_score * 0.15 +
                attack_score * 0.15
            )

            scored_results.append((final_score, result))

        # 按分数排序
        scored_results.sort(reverse=True, key=lambda x: x[0])

        # 返回排序后的结果
        return [result for _, result in scored_results]

    def retrieve_by_attack_type(
        self,
        attack_type: str,
        n_results: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        根据攻击类型检索

        Args:
            attack_type: 攻击类型
            n_results: 返回结果数量

        Returns:
            检索结果列表
        """
        # 获取攻击类型关键词
        keywords = self.ATTACK_KEYWORDS.get(attack_type, [])
        if not keywords:
            logger.warning(f"未知攻击类型: {attack_type}")
            return []

        # 使用关键词构建查询
        query = " ".join(keywords[:3])

        # 检索
        results = self.vector_store.search(
            query=query,
            n_results=n_results,
            where={"attack_type": attack_type},
        )

        return results

    def retrieve_by_risk_level(
        self,
        risk_level: str,
        n_results: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        根据风险等级检索

        Args:
            risk_level: 风险等级（high/medium/low）
            n_results: 返回结果数量

        Returns:
            检索结果列表
        """
        # 风险等级查询映射
        risk_queries = {
            "high": "高危安全事件 威胁 入侵",
            "medium": "中危安全事件 可疑 异常",
            "low": "低危安全事件 信息 建议",
        }

        query = risk_queries.get(risk_level, "安全事件")

        results = self.vector_store.search(
            query=query,
            n_results=n_results,
            where={"risk_level": risk_level},
        )

        return results

    def get_related_knowledge(self, log_text: str) -> str:
        """
        获取与日志相关的安全知识（用于 RAG 增强）

        Args:
            log_text: 日志文本

        Returns:
            相关知识的格式化文本
        """
        # 检测攻击类型
        attack_types = self._detect_attack_type(log_text)

        # 检索相关知识
        all_results = []
        for attack_type in attack_types:
            results = self.retrieve_by_attack_type(attack_type, n_results=2)
            all_results.extend(results)

        # 如果没有检测到攻击类型，使用通用检索
        if not all_results:
            results = self.retrieve(log_text, n_results=3)
            all_results.extend(results)

        # 去重
        seen_contents = set()
        unique_results = []
        for result in all_results:
            content = result.get("content", "")
            if content not in seen_contents:
                seen_contents.add(content)
                unique_results.append(result)

        # 格式化输出
        if not unique_results:
            return "未找到相关安全知识。"

        knowledge_parts = []
        for i, result in enumerate(unique_results[:5], 1):
            content = result.get("content", "")
            metadata = result.get("metadata", {})
            attack_type = metadata.get("attack_type", "未知")
            risk_level = metadata.get("risk_level", "未知")

            knowledge_parts.append(
                f"### 知识 {i}\n"
                f"- 攻击类型: {attack_type}\n"
                f"- 风险等级: {risk_level}\n"
                f"- 内容:\n{content}\n"
            )

        return "\n".join(knowledge_parts)
