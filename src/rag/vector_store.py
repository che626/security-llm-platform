"""
向量数据库存储 - 基于 ChromaDB

功能：
1. 文档向量化存储
2. 语义检索
3. 元数据过滤
4. 持久化存储
"""

import os
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


class VectorStore:
    """
    向量数据库存储

    基于 ChromaDB 实现的向量存储，支持安全知识的语义检索。
    """

    def __init__(
        self,
        collection_name: str = "security_knowledge",
        persist_directory: str = "data/vector_db",
    ):
        """
        初始化向量存储

        Args:
            collection_name: 集合名称
            persist_directory: 持久化目录
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.client = None
        self.collection = None

        # 初始化
        self._init_client()

        logger.info(f"VectorStore 初始化完成")
        logger.info(f"集合: {collection_name}")
        logger.info(f"文档数: {self.count()}")

    def _init_client(self):
        """初始化 ChromaDB 客户端"""
        try:
            import chromadb

            # 创建持久化目录
            os.makedirs(self.persist_directory, exist_ok=True)

            # 初始化客户端
            self.client = chromadb.PersistentClient(path=self.persist_directory)

            # 获取或创建集合
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )

            logger.info(f"ChromaDB 初始化成功")

        except Exception as e:
            logger.error(f"ChromaDB 初始化失败: {e}")
            self.client = None
            self.collection = None

    def add_documents(
        self,
        documents: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
    ) -> List[str]:
        """
        添加文档到向量存储

        Args:
            documents: 文档内容列表
            metadatas: 元数据列表
            ids: 文档 ID 列表

        Returns:
            文档 ID 列表
        """
        if not documents:
            return []

        if self.collection is None:
            logger.error("ChromaDB 未初始化")
            return []

        # 生成 ID
        if ids is None:
            import uuid
            ids = [str(uuid.uuid4()) for _ in range(len(documents))]

        # 生成元数据
        if metadatas is None:
            metadatas = [{} for _ in range(len(documents))]

        # 确保元数据值是字符串
        clean_metadatas = []
        for meta in metadatas:
            clean_meta = {}
            for k, v in meta.items():
                if isinstance(v, (str, int, float, bool)):
                    clean_meta[k] = v
                else:
                    clean_meta[k] = str(v)
            clean_metadatas.append(clean_meta)

        try:
            self.collection.add(
                documents=documents,
                metadatas=clean_metadatas,
                ids=ids,
            )
            logger.info(f"添加了 {len(documents)} 个文档")
            return ids
        except Exception as e:
            logger.error(f"添加文档失败: {e}")
            return []

    def search(
        self,
        query: str,
        n_results: int = 5,
        where: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        语义检索

        Args:
            query: 查询文本
            n_results: 返回结果数量
            where: 元数据过滤条件

        Returns:
            检索结果列表
        """
        if self.collection is None:
            logger.error("ChromaDB 未初始化")
            return []

        try:
            # ChromaDB 检索
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results,
                where=where,
            )

            # 格式化结果
            formatted_results = []
            if results and results["documents"]:
                for i, doc in enumerate(results["documents"][0]):
                    result = {
                        "content": doc,
                        "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                        "id": results["ids"][0][i] if results["ids"] else "",
                        "distance": results["distances"][0][i] if results["distances"] else 0,
                    }
                    formatted_results.append(result)

            logger.info(f"检索到 {len(formatted_results)} 个结果")
            return formatted_results

        except Exception as e:
            logger.error(f"检索失败: {e}")
            return []

    def count(self) -> int:
        """获取文档数量"""
        if self.collection is None:
            return 0
        try:
            return self.collection.count()
        except Exception:
            return 0

    def delete(self, ids: List[str]):
        """删除文档"""
        if self.collection is None:
            return

        try:
            self.collection.delete(ids=ids)
            logger.info(f"删除了 {len(ids)} 个文档")
        except Exception as e:
            logger.error(f"删除文档失败: {e}")

    def clear(self):
        """清空所有文档"""
        if self.client is None:
            return

        try:
            # 删除并重建集合
            self.client.delete_collection(self.collection_name)
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
            logger.info("清空了集合")
        except Exception as e:
            logger.error(f"清空集合失败: {e}")
