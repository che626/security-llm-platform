"""
基于角色的访问控制（RBAC）

功能：
1. 角色定义
2. 权限定义
3. 角色-权限映射
4. 权限检查
"""

import logging
from typing import List, Set, Optional
from enum import Enum

logger = logging.getLogger(__name__)


class Permission(Enum):
    """权限枚举"""
    # 系统管理
    SYSTEM_ADMIN = "system:admin"
    SYSTEM_CONFIG = "system:config"

    # 用户管理
    USER_VIEW = "user:view"
    USER_CREATE = "user:create"
    USER_EDIT = "user:edit"
    USER_DELETE = "user:delete"

    # 分析功能
    ANALYSIS_VIEW = "analysis:view"
    ANALYSIS_EXECUTE = "analysis:execute"

    # 事件管理
    INCIDENT_VIEW = "incident:view"
    INCIDENT_CREATE = "incident:create"
    INCIDENT_EDIT = "incident:edit"
    INCIDENT_DELETE = "incident:delete"

    # 训练功能
    TRAINING_VIEW = "training:view"
    TRAINING_EXECUTE = "training:execute"
    TRAINING_MANAGE = "training:manage"

    # 基准测试
    BENCHMARK_VIEW = "benchmark:view"
    BENCHMARK_EXECUTE = "benchmark:execute"

    # 知识库
    KNOWLEDGE_VIEW = "knowledge:view"
    KNOWLEDGE_EDIT = "knowledge:edit"

    # 报告
    REPORT_VIEW = "report:view"
    REPORT_EXPORT = "report:export"


class Role(Enum):
    """角色枚举"""
    ADMIN = "admin"          # 管理员
    JUDGE = "judge"          # 评审用户
    RESEARCH = "research"    # 研究用户
    USER = "user"            # 普通用户


# 角色-权限映射
ROLE_PERMISSIONS = {
    Role.ADMIN: [
        # 系统管理
        Permission.SYSTEM_ADMIN,
        Permission.SYSTEM_CONFIG,
        # 用户管理
        Permission.USER_VIEW,
        Permission.USER_CREATE,
        Permission.USER_EDIT,
        Permission.USER_DELETE,
        # 分析功能
        Permission.ANALYSIS_VIEW,
        Permission.ANALYSIS_EXECUTE,
        # 事件管理
        Permission.INCIDENT_VIEW,
        Permission.INCIDENT_CREATE,
        Permission.INCIDENT_EDIT,
        Permission.INCIDENT_DELETE,
        # 训练功能
        Permission.TRAINING_VIEW,
        Permission.TRAINING_EXECUTE,
        Permission.TRAINING_MANAGE,
        # 基准测试
        Permission.BENCHMARK_VIEW,
        Permission.BENCHMARK_EXECUTE,
        # 知识库
        Permission.KNOWLEDGE_VIEW,
        Permission.KNOWLEDGE_EDIT,
        # 报告
        Permission.REPORT_VIEW,
        Permission.REPORT_EXPORT,
    ],

    Role.JUDGE: [
        # 分析功能
        Permission.ANALYSIS_VIEW,
        Permission.ANALYSIS_EXECUTE,
        # 事件管理
        Permission.INCIDENT_VIEW,
        Permission.INCIDENT_CREATE,
        Permission.INCIDENT_EDIT,
        # 训练功能
        Permission.TRAINING_VIEW,
        # 基准测试
        Permission.BENCHMARK_VIEW,
        # 知识库
        Permission.KNOWLEDGE_VIEW,
        # 报告
        Permission.REPORT_VIEW,
        Permission.REPORT_EXPORT,
    ],

    Role.RESEARCH: [
        # 分析功能
        Permission.ANALYSIS_VIEW,
        Permission.ANALYSIS_EXECUTE,
        # 事件管理
        Permission.INCIDENT_VIEW,
        # 训练功能
        Permission.TRAINING_VIEW,
        Permission.TRAINING_EXECUTE,
        # 基准测试
        Permission.BENCHMARK_VIEW,
        Permission.BENCHMARK_EXECUTE,
        # 知识库
        Permission.KNOWLEDGE_VIEW,
        Permission.KNOWLEDGE_EDIT,
        # 报告
        Permission.REPORT_VIEW,
        Permission.REPORT_EXPORT,
    ],

    Role.USER: [
        # 分析功能
        Permission.ANALYSIS_VIEW,
        Permission.ANALYSIS_EXECUTE,
        # 事件管理
        Permission.INCIDENT_VIEW,
        # 知识库
        Permission.KNOWLEDGE_VIEW,
        # 报告
        Permission.REPORT_VIEW,
    ],
}


class RBAC:
    """
    基于角色的访问控制

    负责权限检查和角色管理。
    """

    def __init__(self):
        logger.info("RBAC 初始化完成")

    def get_role(self, role_name: str) -> Optional[Role]:
        """
        获取角色枚举

        Args:
            role_name: 角色名称

        Returns:
            角色枚举，不存在返回 None
        """
        try:
            return Role(role_name.lower())
        except ValueError:
            logger.warning(f"未知角色: {role_name}")
            return None

    def get_permissions(self, role: Role) -> List[Permission]:
        """
        获取角色的权限列表

        Args:
            role: 角色

        Returns:
            权限列表
        """
        return ROLE_PERMISSIONS.get(role, [])

    def has_permission(self, role: Role, permission: Permission) -> bool:
        """
        检查角色是否有指定权限

        Args:
            role: 角色
            permission: 权限

        Returns:
            是否有权限
        """
        permissions = self.get_permissions(role)
        return permission in permissions

    def check_permission(
        self,
        user_role: str,
        required_permission: Permission,
    ) -> bool:
        """
        检查用户是否有指定权限

        Args:
            user_role: 用户角色名称
            required_permission: 需要的权限

        Returns:
            是否有权限
        """
        role = self.get_role(user_role)
        if role is None:
            return False

        return self.has_permission(role, required_permission)

    def get_user_permissions(self, user_role: str) -> Set[str]:
        """
        获取用户的所有权限

        Args:
            user_role: 用户角色名称

        Returns:
            权限集合
        """
        role = self.get_role(user_role)
        if role is None:
            return set()

        permissions = self.get_permissions(role)
        return {p.value for p in permissions}

    def can_access_page(self, user_role: str, page_name: str) -> bool:
        """
        检查用户是否可以访问指定页面

        Args:
            user_role: 用户角色名称
            page_name: 页面名称

        Returns:
            是否可以访问
        """
        # 页面-权限映射
        page_permissions = {
            "系统首页": [Permission.ANALYSIS_VIEW],
            "安全态势仪表盘": [Permission.ANALYSIS_VIEW],
            "后端服务状态": [Permission.ANALYSIS_VIEW],
            "AI 安全助手": [Permission.ANALYSIS_VIEW],
            "模型接入配置中心": [Permission.SYSTEM_CONFIG],
            "日志分析器": [Permission.ANALYSIS_VIEW],
            "流量摘要解释器": [Permission.ANALYSIS_VIEW],
            "IOC 威胁指标提取器": [Permission.ANALYSIS_VIEW],
            "攻击链分析": [Permission.ANALYSIS_VIEW],
            "ATT&CK 技术点映射": [Permission.ANALYSIS_VIEW],
            "SOAR 剧本生成器": [Permission.ANALYSIS_EXECUTE],
            "RAG 安全知识库": [Permission.KNOWLEDGE_VIEW],
            "事件处置中心": [Permission.INCIDENT_VIEW],
            "分析历史记录": [Permission.ANALYSIS_VIEW],
            "报告导出中心": [Permission.REPORT_VIEW],
            "安全指令数据集构造器": [Permission.TRAINING_VIEW],
            "DeepSpeed ZeRO 实验展示": [Permission.TRAINING_VIEW],
            "模型效果评测中心": [Permission.TRAINING_VIEW],
            "资产风险画像": [Permission.ANALYSIS_VIEW],
            "安全规则管理": [Permission.SYSTEM_CONFIG],
            "答辩演示中心": [Permission.ANALYSIS_VIEW],
            "系统说明": [Permission.ANALYSIS_VIEW],
        }

        required_permissions = page_permissions.get(page_name, [Permission.ANALYSIS_VIEW])

        # 检查是否有任一所需权限
        for permission in required_permissions:
            if self.check_permission(user_role, permission):
                return True

        return False
