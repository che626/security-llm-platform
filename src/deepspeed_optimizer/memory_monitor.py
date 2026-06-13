"""
GPU 显存实时监控器

功能：
1. 实时采集显存使用数据
2. 预测 OOM 风险
3. 提供显存使用趋势分析
"""

import torch
import threading
import time
from collections import deque
from typing import Dict, Any, Optional, List
import logging

logger = logging.getLogger(__name__)


class MemoryMonitor:
    """
    GPU 显存实时监控器

    核心创新：
    1. 后台线程持续采集显存数据（每 100ms）
    2. 基于线性回归预测 OOM 风险
    3. 提供显存使用趋势分析
    """

    def __init__(self, interval: float = 0.1, history_size: int = 1000):
        """
        初始化监控器

        Args:
            interval: 采集间隔（秒）
            history_size: 历史记录最大数量
        """
        self.interval = interval
        self.history = deque(maxlen=history_size)
        self.running = False
        self.thread = None
        self.lock = threading.Lock()

        # GPU 信息
        if torch.cuda.is_available():
            self.total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        else:
            self.total_memory = 0

        logger.info(f"MemoryMonitor 初始化，GPU 显存: {self.total_memory:.2f} GB")

    def start(self):
        """开始监控"""
        if self.running:
            return

        self.running = True
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()
        logger.info("显存监控已启动")

    def stop(self):
        """停止监控"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=1)
        logger.info("显存监控已停止")

    def _monitor_loop(self):
        """监控循环"""
        while self.running:
            if torch.cuda.is_available():
                with self.lock:
                    self.history.append({
                        "time": time.time(),
                        "allocated": torch.cuda.memory_allocated() / 1024**3,
                        "reserved": torch.cuda.memory_reserved() / 1024**3,
                        "max_allocated": torch.cuda.max_memory_allocated() / 1024**3,
                    })
            time.sleep(self.interval)

    def get_current_usage(self) -> float:
        """获取当前显存使用率 (0-1)"""
        if not torch.cuda.is_available():
            return 0
        return torch.cuda.memory_allocated() / torch.cuda.get_device_properties(0).total_memory

    def get_current_allocated_gb(self) -> float:
        """获取当前已分配显存 (GB)"""
        if not torch.cuda.is_available():
            return 0
        return torch.cuda.memory_allocated() / 1024**3

    def get_peak_usage(self) -> float:
        """获取峰值显存使用率 (0-1)"""
        if not self.history:
            return self.get_current_usage()

        with self.lock:
            max_allocated = max(h["max_allocated"] for h in self.history)

        return max_allocated / self.total_memory if self.total_memory > 0 else 0

    def is_memory_critical(self, threshold: float = 0.85) -> bool:
        """检查显存是否接近上限"""
        return self.get_current_usage() > threshold

    def predict_oom(self, threshold: float = 0.95, steps_ahead: int = 10) -> bool:
        """
        预测是否会 OOM

        使用简单线性回归预测未来显存使用

        Args:
            threshold: OOM 阈值
            steps_ahead: 预测未来几步

        Returns:
            是否会 OOM
        """
        with self.lock:
            if len(self.history) < 10:
                return False

            # 取最近 10 个采样点
            recent = list(self.history)[-10:]

        # 简单线性回归
        n = len(recent)
        x = list(range(n))
        y = [h["allocated"] for h in recent]

        x_mean = sum(x) / n
        y_mean = sum(y) / n

        numerator = sum((xi - x_mean) * (yi - y_mean) for xi, yi in zip(x, y))
        denominator = sum((xi - x_mean) ** 2 for xi in x)

        if denominator == 0:
            return False

        slope = numerator / denominator
        intercept = y_mean - slope * x_mean

        # 预测
        predicted = slope * (n + steps_ahead) + intercept

        return predicted / self.total_memory > threshold

    def get_memory_trend(self) -> str:
        """获取显存使用趋势"""
        with self.lock:
            if len(self.history) < 10:
                return "insufficient_data"

            recent = list(self.history)[-10:]

        # 计算趋势
        first_half = sum(h["allocated"] for h in recent[:5]) / 5
        second_half = sum(h["allocated"] for h in recent[5:]) / 5

        if second_half > first_half * 1.1:
            return "increasing"
        elif second_half < first_half * 0.9:
            return "decreasing"
        else:
            return "stable"

    def get_statistics(self) -> Dict[str, Any]:
        """获取显存统计信息"""
        with self.lock:
            if not self.history:
                return {
                    "current_gb": self.get_current_allocated_gb(),
                    "peak_gb": self.get_current_allocated_gb(),
                    "avg_gb": self.get_current_allocated_gb(),
                    "usage_percent": self.get_current_usage() * 100,
                    "trend": "unknown",
                    "samples": 0,
                }

            allocated_values = [h["allocated"] for h in self.history]

        return {
            "current_gb": round(allocated_values[-1], 3),
            "peak_gb": round(max(allocated_values), 3),
            "avg_gb": round(sum(allocated_values) / len(allocated_values), 3),
            "usage_percent": round(self.get_current_usage() * 100, 2),
            "trend": self.get_memory_trend(),
            "samples": len(self.history),
        }

    def get_history(self, last_n: Optional[int] = None) -> List[Dict[str, Any]]:
        """获取历史数据"""
        with self.lock:
            if last_n:
                return list(self.history)[-last_n:]
            return list(self.history)
