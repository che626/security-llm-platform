"""
自适应批大小调整器

核心创新：
根据 GPU 显存使用情况动态调整 batch size，
最大化显存利用率的同时避免 OOM。
"""

import logging
from typing import List, Dict, Any
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class BatchAdjustmentRecord:
    """批大小调整记录"""
    step: int
    memory_usage: float
    old_batch_size: int
    new_batch_size: int
    reason: str


class AdaptiveBatchSizer:
    """
    自适应批大小调整器

    核心策略：
    1. 显存使用 < 60%：增大 batch size（快速利用空闲显存）
    2. 显存使用 60-85%：保持不变（安全区间）
    3. 显存使用 > 85%：减小 batch size（预防 OOM）
    4. OOM 发生：大幅减半 batch size（紧急降级）
    """

    def __init__(
        self,
        initial_batch_size: int = 4,
        min_batch_size: int = 1,
        max_batch_size: int = 32,
        safe_threshold: float = 0.60,
        warning_threshold: float = 0.85,
        critical_threshold: float = 0.95,
    ):
        """
        初始化

        Args:
            initial_batch_size: 初始批大小
            min_batch_size: 最小批大小
            max_batch_size: 最大批大小
            safe_threshold: 安全阈值（低于此值可增大）
            warning_threshold: 警告阈值（高于此值需减小）
            critical_threshold: 临界阈值（紧急减小）
        """
        self.current_batch_size = initial_batch_size
        self.initial_batch_size = initial_batch_size
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.safe_threshold = safe_threshold
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold

        # 调整历史
        self.history: List[BatchAdjustmentRecord] = []
        self.step = 0

        # 统计
        self.total_adjustments = 0
        self.oom_count = 0

        logger.info(f"AdaptiveBatchSizer 初始化: batch_size={initial_batch_size}")

    def adjust(self, memory_usage: float, oom_occurred: bool = False) -> int:
        """
        根据显存使用调整批大小

        Args:
            memory_usage: 当前显存使用率 (0-1)
            oom_occurred: 是否发生了 OOM

        Returns:
            新的批大小
        """
        self.step += 1
        old_batch_size = self.current_batch_size

        if oom_occurred:
            # OOM 发生，大幅减半
            self.oom_count += 1
            self.current_batch_size = max(
                self.min_batch_size,
                self.current_batch_size // 2
            )
            reason = f"OOM 发生 (第 {self.oom_count} 次)"

        elif memory_usage > self.critical_threshold:
            # 临界状态，减半
            self.current_batch_size = max(
                self.min_batch_size,
                self.current_batch_size // 2
            )
            reason = "显存临界"

        elif memory_usage > self.warning_threshold:
            # 警告状态，减小 1
            self.current_batch_size = max(
                self.min_batch_size,
                self.current_batch_size - 1
            )
            reason = "显存紧张"

        elif memory_usage < self.safe_threshold:
            # 安全状态，尝试增大
            self.current_batch_size = min(
                self.max_batch_size,
                self.current_batch_size + 1
            )
            reason = "显存充足，增大 batch"

        else:
            # 保持不变
            reason = "显存正常"

        # 记录调整
        if self.current_batch_size != old_batch_size:
            self.total_adjustments += 1
            record = BatchAdjustmentRecord(
                step=self.step,
                memory_usage=memory_usage,
                old_batch_size=old_batch_size,
                new_batch_size=self.current_batch_size,
                reason=reason,
            )
            self.history.append(record)

            logger.info(
                f"批大小调整: {old_batch_size} -> {self.current_batch_size} "
                f"(显存: {memory_usage:.1%}, 原因: {reason})"
            )

        return self.current_batch_size

    def get_current_batch_size(self) -> int:
        """获取当前批大小"""
        return self.current_batch_size

    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "current_batch_size": self.current_batch_size,
            "initial_batch_size": self.initial_batch_size,
            "total_adjustments": self.total_adjustments,
            "oom_count": self.oom_count,
            "history_length": len(self.history),
        }

    def get_history(self) -> List[Dict[str, Any]]:
        """获取调整历史"""
        return [
            {
                "step": r.step,
                "memory_usage": round(r.memory_usage, 3),
                "old_batch_size": r.old_batch_size,
                "new_batch_size": r.new_batch_size,
                "reason": r.reason,
            }
            for r in self.history
        ]

    def reset(self):
        """重置为初始状态"""
        self.current_batch_size = self.initial_batch_size
        self.history.clear()
        self.total_adjustments = 0
        self.oom_count = 0
        self.step = 0
        logger.info("AdaptiveBatchSizer 已重置")
