"""
健康检查模块

功能：
1. 系统健康检查
2. 服务依赖检查
3. 健康状态报告
"""

import time
import logging
from typing import Dict, Any, List, Callable
from datetime import datetime

logger = logging.getLogger(__name__)


class HealthChecker:
    """
    健康检查器

    检查系统和服务的健康状态。
    """

    def __init__(self):
        """初始化健康检查器"""
        self.checks: Dict[str, Callable] = {}
        self.register_default_checks()
        logger.info("HealthChecker 初始化完成")

    def register_default_checks(self):
        """注册默认检查项"""
        self.register_check("database", self._check_database)
        self.register_check("gpu", self._check_gpu)
        self.register_check("disk_space", self._check_disk_space)
        self.register_check("memory", self._check_memory)

    def register_check(self, name: str, check_func: Callable):
        """
        注册检查项

        Args:
            name: 检查项名称
            check_func: 检查函数
        """
        self.checks[name] = check_func

    def _check_database(self) -> Dict[str, Any]:
        """检查数据库连接"""
        try:
            from src.database.session import get_session
            session = get_session()
            session.execute("SELECT 1")
            session.close()
            return {"status": "healthy", "message": "数据库连接正常"}
        except Exception as e:
            return {"status": "unhealthy", "message": f"数据库连接失败: {e}"}

    def _check_gpu(self) -> Dict[str, Any]:
        """检查 GPU 状态"""
        try:
            import torch
            if not torch.cuda.is_available():
                return {"status": "warning", "message": "GPU 不可用"}

            gpu_count = torch.cuda.device_count()
            memory_allocated = torch.cuda.memory_allocated() / (1024 ** 3)
            memory_total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            memory_usage = memory_allocated / memory_total * 100

            status = "healthy" if memory_usage < 90 else "warning"

            return {
                "status": status,
                "message": f"GPU 数量: {gpu_count}, 显存使用: {memory_usage:.1f}%",
                "gpu_count": gpu_count,
                "memory_allocated_gb": round(memory_allocated, 2),
                "memory_total_gb": round(memory_total, 2),
                "memory_usage_percent": round(memory_usage, 1),
            }
        except ImportError:
            return {"status": "warning", "message": "PyTorch 未安装"}
        except Exception as e:
            return {"status": "unhealthy", "message": f"GPU 检查失败: {e}"}

    def _check_disk_space(self) -> Dict[str, Any]:
        """检查磁盘空间"""
        try:
            import psutil
            disk = psutil.disk_usage("/")
            usage_percent = disk.percent

            status = "healthy" if usage_percent < 80 else "warning" if usage_percent < 90 else "unhealthy"

            return {
                "status": status,
                "message": f"磁盘使用率: {usage_percent:.1f}%",
                "total_gb": round(disk.total / (1024 ** 3), 2),
                "used_gb": round(disk.used / (1024 ** 3), 2),
                "free_gb": round(disk.free / (1024 ** 3), 2),
                "usage_percent": round(usage_percent, 1),
            }
        except ImportError:
            return {"status": "warning", "message": "psutil 未安装"}
        except Exception as e:
            return {"status": "unhealthy", "message": f"磁盘检查失败: {e}"}

    def _check_memory(self) -> Dict[str, Any]:
        """检查内存状态"""
        try:
            import psutil
            memory = psutil.virtual_memory()
            usage_percent = memory.percent

            status = "healthy" if usage_percent < 80 else "warning" if usage_percent < 90 else "unhealthy"

            return {
                "status": status,
                "message": f"内存使用率: {usage_percent:.1f}%",
                "total_gb": round(memory.total / (1024 ** 3), 2),
                "used_gb": round(memory.used / (1024 ** 3), 2),
                "available_gb": round(memory.available / (1024 ** 3), 2),
                "usage_percent": round(usage_percent, 1),
            }
        except ImportError:
            return {"status": "warning", "message": "psutil 未安装"}
        except Exception as e:
            return {"status": "unhealthy", "message": f"内存检查失败: {e}"}

    def run_check(self, name: str) -> Dict[str, Any]:
        """
        运行指定检查

        Args:
            name: 检查项名称

        Returns:
            检查结果
        """
        check_func = self.checks.get(name)
        if not check_func:
            return {"status": "unknown", "message": f"未知检查项: {name}"}

        start_time = time.time()
        try:
            result = check_func()
            duration = time.time() - start_time
            result["duration_ms"] = round(duration * 1000, 2)
            result["check_name"] = name
            result["timestamp"] = datetime.utcnow().isoformat()
            return result
        except Exception as e:
            return {
                "status": "error",
                "message": f"检查执行失败: {e}",
                "check_name": name,
                "timestamp": datetime.utcnow().isoformat(),
            }

    def run_all_checks(self) -> Dict[str, Any]:
        """
        运行所有检查

        Returns:
            所有检查结果
        """
        results = {}
        overall_status = "healthy"

        for name in self.checks:
            result = self.run_check(name)
            results[name] = result

            # 更新整体状态
            if result["status"] == "unhealthy" or result["status"] == "error":
                overall_status = "unhealthy"
            elif result["status"] == "warning" and overall_status == "healthy":
                overall_status = "warning"

        return {
            "status": overall_status,
            "timestamp": datetime.utcnow().isoformat(),
            "checks": results,
        }

    def get_health_summary(self) -> str:
        """获取健康状态摘要"""
        all_checks = self.run_all_checks()

        lines = [
            f"# 系统健康状态: {all_checks['status'].upper()}",
            f"检查时间: {all_checks['timestamp']}",
            "",
            "## 检查详情",
        ]

        for name, result in all_checks["checks"].items():
            status_icon = {
                "healthy": "✅",
                "warning": "⚠️",
                "unhealthy": "❌",
                "error": "💥",
                "unknown": "❓",
            }.get(result["status"], "❓")

            lines.append(f"- {status_icon} **{name}**: {result['message']}")
            if "duration_ms" in result:
                lines.append(f"  - 耗时: {result['duration_ms']}ms")

        return "\n".join(lines)
