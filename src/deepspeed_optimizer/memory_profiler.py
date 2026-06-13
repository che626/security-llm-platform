"""
显存分析器 - 用于分析和优化 GPU 显存使用

功能：
1. 实时显存监控
2. 显存泄漏检测
3. 显存使用热点分析
4. 优化建议生成
"""

import time
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from collections import deque

logger = logging.getLogger(__name__)


@dataclass
class MemorySnapshot:
    """显存快照"""
    timestamp: float
    allocated_gb: float
    reserved_gb: float
    max_allocated_gb: float
    total_gb: float
    fragmentation_gb: float
    step: int = 0
    label: str = ""

    @property
    def utilization(self) -> float:
        """显存利用率"""
        if self.total_gb == 0:
            return 0.0
        return self.allocated_gb / self.total_gb * 100

    @property
    def fragmentation_ratio(self) -> float:
        """碎片率"""
        if self.reserved_gb == 0:
            return 0.0
        return self.fragmentation_gb / self.reserved_gb * 100

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "allocated_gb": round(self.allocated_gb, 3),
            "reserved_gb": round(self.reserved_gb, 3),
            "max_allocated_gb": round(self.max_allocated_gb, 3),
            "total_gb": round(self.total_gb, 3),
            "fragmentation_gb": round(self.fragmentation_gb, 3),
            "utilization_percent": round(self.utilization, 2),
            "fragmentation_percent": round(self.fragmentation_ratio, 2),
            "step": self.step,
            "label": self.label,
        }


@dataclass
class MemoryAnalysis:
    """显存分析结果"""
    peak_memory_gb: float
    avg_memory_gb: float
    min_memory_gb: float
    memory_growth_rate: float  # GB/step
    fragmentation_avg: float
    leak_suspected: bool
    leak_confidence: float  # 0-1
    recommendations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "peak_memory_gb": round(self.peak_memory_gb, 3),
            "avg_memory_gb": round(self.avg_memory_gb, 3),
            "min_memory_gb": round(self.min_memory_gb, 3),
            "memory_growth_rate": round(self.memory_growth_rate, 6),
            "fragmentation_avg_percent": round(self.fragmentation_avg, 2),
            "leak_suspected": self.leak_suspected,
            "leak_confidence": round(self.leak_confidence, 3),
            "recommendations": self.recommendations,
        }


class MemoryProfiler:
    """
    显存分析器

    实时监控 GPU 显存使用，检测内存泄漏，生成优化建议。
    """

    def __init__(self, max_snapshots: int = 1000):
        self.snapshots: deque = deque(maxlen=max_snapshots)
        self.is_monitoring = False
        self._monitor_thread = None

        # 泄漏检测阈值
        self.leak_threshold_gb = 0.1  # 每步增长超过 0.1GB 可能是泄漏
        self.leak_window_size = 50  # 检测窗口大小

        logger.info("MemoryProfiler 初始化完成")

    def take_snapshot(self, step: int = 0, label: str = "") -> Optional[MemorySnapshot]:
        """获取当前显存快照"""
        try:
            import torch
            if not torch.cuda.is_available():
                return None

            allocated = torch.cuda.memory_allocated() / (1024 ** 3)
            reserved = torch.cuda.memory_reserved() / (1024 ** 3)
            max_allocated = torch.cuda.max_memory_allocated() / (1024 ** 3)
            total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            fragmentation = reserved - allocated

            snapshot = MemorySnapshot(
                timestamp=time.time(),
                allocated_gb=allocated,
                reserved_gb=reserved,
                max_allocated_gb=max_allocated,
                total_gb=total,
                fragmentation_gb=fragmentation,
                step=step,
                label=label,
            )

            self.snapshots.append(snapshot)
            return snapshot

        except Exception as e:
            logger.error(f"获取显存快照失败: {e}")
            return None

    def analyze(self) -> Optional[MemoryAnalysis]:
        """分析显存使用情况"""
        if len(self.snapshots) < 10:
            logger.warning("快照数量不足，无法进行分析")
            return None

        # 提取数据
        memory_values = [s.allocated_gb for s in self.snapshots]
        fragmentation_values = [s.fragmentation_ratio for s in self.snapshots]

        # 基本统计
        peak_memory = max(memory_values)
        avg_memory = sum(memory_values) / len(memory_values)
        min_memory = min(memory_values)
        avg_fragmentation = sum(fragmentation_values) / len(fragmentation_values)

        # 内存增长分析（线性回归）
        n = len(memory_values)
        x_mean = (n - 1) / 2
        y_mean = avg_memory

        numerator = sum((i - x_mean) * (y - y_mean) for i, y in enumerate(memory_values))
        denominator = sum((i - x_mean) ** 2 for i in range(n))

        growth_rate = numerator / denominator if denominator != 0 else 0

        # 泄漏检测
        leak_suspected = False
        leak_confidence = 0.0

        if growth_rate > self.leak_threshold_gb:
            leak_suspected = True
            leak_confidence = min(1.0, growth_rate / (self.leak_threshold_gb * 5))

        # 生成建议
        recommendations = self._generate_recommendations(
            peak_memory, avg_memory, avg_fragmentation, growth_rate, leak_suspected
        )

        return MemoryAnalysis(
            peak_memory_gb=peak_memory,
            avg_memory_gb=avg_memory,
            min_memory_gb=min_memory,
            memory_growth_rate=growth_rate,
            fragmentation_avg=avg_fragmentation,
            leak_suspected=leak_suspected,
            leak_confidence=leak_confidence,
            recommendations=recommendations,
        )

    def _generate_recommendations(
        self,
        peak_memory: float,
        avg_memory: float,
        avg_fragmentation: float,
        growth_rate: float,
        leak_suspected: bool,
    ) -> List[str]:
        """生成优化建议"""
        recommendations = []

        # 显存使用建议
        if peak_memory > 0.9 * self.snapshots[-1].total_gb:
            recommendations.append(
                "显存使用率超过 90%，建议启用 ZeRO-3 或 CPU Offload 以减少显存压力"
            )

        if avg_memory < 0.5 * self.snapshots[-1].total_gb:
            recommendations.append(
                "显存使用率低于 50%，可以考虑增大 batch size 或减少 ZeRO Stage 以提升训练速度"
            )

        # 碎片化建议
        if avg_fragmentation > 20:
            recommendations.append(
                f"显存碎片率较高 ({avg_fragmentation:.1f}%)，建议定期调用 torch.cuda.empty_cache()"
            )

        # 增长率建议
        if growth_rate > 0:
            recommendations.append(
                f"显存持续增长 ({growth_rate*1000:.2f} MB/step)，可能存在内存泄漏"
            )

        # 泄漏建议
        if leak_suspected:
            recommendations.append(
                "检测到可能的内存泄漏，建议检查：1) 是否有张量未释放 2) 是否有循环引用 3) 是否有未关闭的文件句柄"
            )

        # 通用建议
        if not recommendations:
            recommendations.append("显存使用正常，无需特别优化")

        return recommendations

    def get_memory_timeline(self) -> List[Dict[str, Any]]:
        """获取显存时间线数据（用于绘图）"""
        return [
            {
                "step": s.step,
                "allocated_gb": round(s.allocated_gb, 3),
                "reserved_gb": round(s.reserved_gb, 3),
                "timestamp": s.timestamp,
            }
            for s in self.snapshots
        ]

    def clear(self):
        """清空快照"""
        self.snapshots.clear()
        logger.info("显存快照已清空")

    def reset_peak_memory(self):
        """重置峰值显存统计"""
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
                logger.info("峰值显存统计已重置")
        except Exception as e:
            logger.error(f"重置峰值显存失败: {e}")
