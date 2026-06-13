"""
RAG 向量检索测试脚本

用法：
    python training/test_rag.py
"""

import os
import sys

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.rag import KnowledgeManager


def main():
    """主函数"""
    print("=" * 60)
    print("RAG 向量检索系统测试")
    print("=" * 60)

    # 初始化知识库管理器
    print("\n初始化知识库管理器...")
    km = KnowledgeManager(persist_directory="data/vector_db")

    # 导入知识
    print("\n导入安全知识库...")
    total = km.import_all_knowledge()
    print(f"导入完成: {total} 条知识")

    # 获取统计信息
    stats = km.get_statistics()
    print(f"\n知识库统计:")
    print(f"  文档数: {stats['total_knowledge']}")
    print(f"  存储类型: {stats['vector_store_type']}")

    # 测试检索
    test_queries = [
        "SSH 登录失败怎么处理？",
        "Web 日志出现 SQL 注入怎么办？",
        "发现可疑 C2 回连流量",
        "服务器被 DDoS 攻击了",
        "员工在下载敏感数据",
    ]

    print("\n" + "=" * 60)
    print("检索测试")
    print("=" * 60)

    for query in test_queries:
        print(f"\n查询: {query}")
        print("-" * 40)

        results = km.search(query, n_results=2)

        if results:
            for i, result in enumerate(results, 1):
                content = result.get("content", "")[:200]
                metadata = result.get("metadata", {})
                attack_type = metadata.get("attack_type", "未知")
                risk_level = metadata.get("risk_level", "未知")

                print(f"  结果 {i}:")
                print(f"    攻击类型: {attack_type}")
                print(f"    风险等级: {risk_level}")
                print(f"    内容: {content}...")
        else:
            print("  未找到相关结果")

    # 测试 RAG 增强
    print("\n" + "=" * 60)
    print("RAG 增强测试")
    print("=" * 60)

    log_text = "May 11 10:01:02 server sshd[1001]: Failed password for root from 8.8.8.8 port 53422 ssh2"
    print(f"\n日志: {log_text}")
    print("-" * 40)

    knowledge = km.get_related_knowledge(log_text, n_results=2)
    print(f"\n相关知识:\n{knowledge}")

    print("\n" + "=" * 60)
    print("测试完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
