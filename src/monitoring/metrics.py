"""
性能指标收集模块

功能：
1. 系统指标收集（CPU、内存、GPU）
2. 应用指标收集（请求数、响应时间）
3. 指标导出
"""

import time
import logging
from typing import Dict, Any, Optional
from collections import defaultdict, deque
from datetime import datetime

logger = logging.getLogger(__name__)


class MetricsCollector:
    """
    性能指标收集器

    收集系统和应用的性能指标。
    """

    def __init__(self, max_history: int = 1000):
        """
        初始化指标收集器

        Args:
            max_history: 最大历史记录数
        """
        self.max_history = max_history

        # 计数器
        self.counters = defaultdict(int)

        # 计时器
        self.timers = defaultdict(list)

        # 历史记录
        self.history = deque(maxlen=max_history)

        logger.info("MetricsCollector 初始化完成")

    def increment(self, name: str, value: int = 1):
        """
        增加计数器

        Args:
            name: 计数器名称
            value: 增加值
        """
        self.counters[name] += value

    def decrement(self, name: str, value: int = 1):
        """
        减少计数器

        Args:
            name: 计数器名称
            value: 减少值
        """
        self.counters[name] -= value

    def set_gauge(self, name: str, value: float):
        """
        设置仪表值

        Args:
            name: 仪表名称
            value: 仪表值
        """
        self.counters[name] = value

    def record_time(self, name: str, duration: float):
        """
        记录时间

        Args:
            name: 计时器名称
            duration: 持续时间（秒）
        """
        self.timers[name].append(duration)

        # 保持最近的记录
        if len(self.timers[name]) > self.max_history:
            self.timers[name] = self.timers[name][-self.max_history:]

    def record_metric(self, name: str, value: Any, tags: Optional[Dict[str, str]] = None):
        """
        记录指标

        Args:
            name: 指标名称
            value: 指标值
            tags: 标签
        """
        metric = {
            "name": name,
            "value": value,
            "tags": tags or {},
            "timestamp": datetime.utcnow().isoformat(),
        }
        self.history.append(metric)

    def get_counter(self, name: str) -> int:
        """获取计数器值"""
        return self.counters.get(name, 0)

    def get_timer_stats(self, name: str) -> Dict[str, float]:
        """获取计时器统计"""
        times = self.timers.get(name, [])
        if not times:
            return {"count": 0, "avg": 0, "min": 0, "max": 0, "total": 0}

        return {
            "count": len(times),
            "avg": sum(times) / len(times),
            "min": min(times),
            "max": max(times),
            "total": sum(times),
        }

    def get_system_metrics(self) -> Dict[str, Any]:
        """获取系统指标"""
        metrics = {}

        try:
            import psutil

            # CPU 指标
            metrics["cpu_percent"] = psutil.cpu_percent(interval=1)
            metrics["cpu_count"] = psutil.cpu_count()

            # 内存指标
            memory = psutil.virtual_memory()
            metrics["memory_total_gb"] = memory.total / (1024 ** 3)
            metrics["memory_used_gb"] = memory.used / (1024 ** 3)
            metrics["memory_percent"] = memory.percent

            # 磁盘指标
            disk = psutil.disk_usage("/")
            metrics["disk_total_gb"] = disk.total / (1024 ** 3)
            metrics["disk_used_gb"] = disk.used / (1024 ** 3)
            metrics["disk_percent"] = disk.percent

        except ImportError:
            logger.warning("psutil 未安装，无法获取系统指标")

        # GPU 指标
        try:
            import torch
            if torch.cuda.is_available():
                metrics["gpu_count"] = torch.cuda.device_count()
                metrics["gpu_memory_allocated_gb"] = torch.cuda.memory_allocated() / (1024 ** 3)
                metrics["gpu_memory_reserved_gb"] = torch.cuda.memory_reserved() / (1024 ** 3)
                metrics["gpu_memory_total_gb"] = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        except ImportError:
            pass

        return metrics

    def get_all_metrics(self) -> Dict[str, Any]:
        """获取所有指标"""
        return {
            "counters": dict(self.counters),
            "timers": {name: self.get_timer_stats(name) for name in self.timers},
            "system": self.get_system_metrics(),
            "history_count": len(self.history),
        }

    def reset(self):
        """重置所有指标"""
        self.counters.clear()
        self.timers.clear()
        self.history.clear()
        logger.info("指标已重置")

    def export_prometheus(self) -> str:
        """导出 Prometheus 格式指标"""
        lines = []

        # 计数器
        for name, value in self.counters.items():
            lines.append(f"# TYPE {name} counter")
            lines.append(f"{name} {value}")

        # 计时器
        for name, stats in self.get_timer_stats(name).items():
            lines.append(f"# TYPE {name}_avg gauge")
            lines.append(f"{name}_avg {stats['avg']}")
            lines.append(f"# TYPE {name}_count counter")
            lines.append(f"{name}_count {stats['count']}")

        return "\n".join(lines)
