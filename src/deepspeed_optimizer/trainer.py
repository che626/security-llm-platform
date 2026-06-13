"""
内存感知训练器 - 核心创新模块

核心创新：
1. 实时监控 GPU 显存
2. 动态调整 batch size
3. 智能选择检查点层
4. 预测并预防 OOM

对比原生训练：
- 原生：显存不足时直接 OOM 崩溃
- 我们：提前预警，动态降级，保证训练继续
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import time
import json
import os
import logging
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field

from .memory_monitor import MemoryMonitor
from .adaptive_batch import AdaptiveBatchSizer
from .smart_checkpoint import SmartCheckpointManager

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """训练配置"""
    model_name: str = "custom"
    output_dir: str = "outputs/training"
    num_epochs: int = 3
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_steps: int = 100
    max_seq_length: int = 512
    initial_batch_size: int = 4
    gradient_accumulation_steps: int = 1
    gradient_clipping: float = 1.0
    fp16: bool = True
    save_steps: int = 500
    logging_steps: int = 10

    # 内存感知配置
    enable_memory_aware: bool = True
    memory_safe_threshold: float = 0.60
    memory_warning_threshold: float = 0.85
    memory_critical_threshold: float = 0.95
    enable_smart_checkpoint: bool = True
    enable_adaptive_batch: bool = True


@dataclass
class TrainingMetrics:
    """训练指标"""
    step: int = 0
    epoch: int = 0
    loss: float = 0.0
    learning_rate: float = 0.0
    batch_size: int = 0
    memory_usage: float = 0.0
    memory_allocated_gb: float = 0.0
    step_time: float = 0.0
    throughput: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step": self.step,
            "epoch": self.epoch,
            "loss": round(self.loss, 6),
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "memory_usage": round(self.memory_usage, 4),
            "memory_allocated_gb": round(self.memory_allocated_gb, 3),
            "step_time": round(self.step_time, 4),
            "throughput": round(self.throughput, 2),
        }


class MemoryAwareTrainer:
    """
    内存感知训练器

    核心功能：
    1. 实时监控显存使用
    2. 动态调整 batch size
    3. 智能选择检查点层
    4. 预测并预防 OOM
    5. 记录完整训练指标
    """

    def __init__(
        self,
        model: nn.Module,
        train_dataset: Dataset,
        val_dataset: Optional[Dataset] = None,
        config: Optional[TrainingConfig] = None,
    ):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.config = config or TrainingConfig()

        # 设备
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 内存感知组件
        self.memory_monitor = MemoryMonitor()
        self.batch_sizer = AdaptiveBatchSizer(
            initial_batch_size=self.config.initial_batch_size,
            safe_threshold=self.config.memory_safe_threshold,
            warning_threshold=self.config.memory_warning_threshold,
            critical_threshold=self.config.memory_critical_threshold,
        )
        self.checkpoint_manager = SmartCheckpointManager(
            memory_threshold_critical=self.config.memory_critical_threshold,
            memory_threshold_normal=self.config.memory_warning_threshold,
        )

        # 训练状态
        self.global_step = 0
        self.current_epoch = 0
        self.metrics_history: List[TrainingMetrics] = []

        # 优化器（延迟初始化）
        self.optimizer = None
        self.scheduler = None

        logger.info("MemoryAwareTrainer 初始化完成")
        logger.info(f"  模型: {self.config.model_name}")
        logger.info(f"  设备: {self.device}")
        logger.info(f"  初始 batch size: {self.config.initial_batch_size}")
        logger.info(f"  内存感知: {self.config.enable_memory_aware}")
        logger.info(f"  智能检查点: {self.config.enable_smart_checkpoint}")
        logger.info(f"  自适应批大小: {self.config.enable_adaptive_batch}")

    def _init_optimizer(self):
        """初始化优化器"""
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

    def _create_dataloader(self, batch_size: int) -> DataLoader:
        """创建数据加载器"""
        return DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True,
        )

    def _train_step(self, batch: Dict[str, torch.Tensor]) -> float:
        """单步训练"""
        # 移动数据到设备
        batch = {k: v.to(self.device) for k, v in batch.items()}

        # 前向传播
        outputs = self.model(**batch)
        loss = outputs.loss if hasattr(outputs, 'loss') else outputs[0]

        # 梯度累积
        loss = loss / self.config.gradient_accumulation_steps

        # 反向传播
        loss.backward()

        return loss.item() * self.config.gradient_accumulation_steps

    def _optimize_step(self):
        """优化器步骤"""
        # 梯度裁剪
        if self.config.gradient_clipping > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.gradient_clipping,
            )

        # 优化器步骤
        self.optimizer.step()

        # 学习率调度
        if self.scheduler:
            self.scheduler.step()

        # 清零梯度
        self.optimizer.zero_grad()

    def _handle_oom(self):
        """处理 OOM"""
        logger.warning("OOM 发生，执行紧急降级")

        # 清理显存
        torch.cuda.empty_cache()

        # 减小批大小
        self.batch_sizer.adjust(1.0, oom_occurred=True)

        # 重新初始化优化器（清除可能的损坏状态）
        self._init_optimizer()

    def train(self) -> List[TrainingMetrics]:
        """
        执行训练

        Returns:
            训练指标历史
        """
        logger.info("=" * 60)
        logger.info("开始内存感知训练")
        logger.info("=" * 60)

        # 移动模型到设备
        self.model = self.model.to(self.device)

        # 初始化优化器
        self._init_optimizer()

        # 应用智能检查点
        if self.config.enable_smart_checkpoint:
            memory_usage = self.memory_monitor.get_current_usage()
            self.checkpoint_manager.apply_smart_checkpointing(self.model, memory_usage)

        # 启动显存监控
        if self.config.enable_memory_aware:
            self.memory_monitor.start()

        # 训练循环
        self.model.train()

        for epoch in range(self.config.num_epochs):
            self.current_epoch = epoch
            epoch_start = time.time()

            logger.info(f"\nEpoch {epoch + 1}/{self.config.num_epochs}")

            # 获取当前 batch size
            current_batch_size = self.batch_sizer.get_current_batch_size()
            dataloader = self._create_dataloader(current_batch_size)

            epoch_loss = 0.0
            epoch_steps = 0

            for batch_idx, batch in enumerate(dataloader):
                step_start = time.time()

                try:
                    # 训练一步
                    loss = self._train_step(batch)

                    # 梯度累积后优化
                    if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                        self._optimize_step()
                        self.global_step += 1

                        # 记录指标
                        step_time = time.time() - step_start
                        memory_stats = self.memory_monitor.get_statistics()

                        metrics = TrainingMetrics(
                            step=self.global_step,
                            epoch=epoch,
                            loss=loss,
                            learning_rate=self.optimizer.param_groups[0]['lr'],
                            batch_size=current_batch_size,
                            memory_usage=memory_stats.get('usage_percent', 0) / 100,
                            memory_allocated_gb=memory_stats.get('current_gb', 0),
                            step_time=step_time,
                            throughput=1 / step_time if step_time > 0 else 0,
                        )
                        self.metrics_history.append(metrics)

                        epoch_loss += loss
                        epoch_steps += 1

                        # 内存感知调整
                        if self.config.enable_memory_aware and self.config.enable_adaptive_batch:
                            memory_usage = self.memory_monitor.get_current_usage()

                            # 预测 OOM
                            if self.memory_monitor.predict_oom():
                                logger.warning("预测到 OOM 风险，提前调整")
                                self.batch_sizer.adjust(memory_usage)
                                # 需要重建 dataloader
                                new_batch_size = self.batch_sizer.get_current_batch_size()
                                if new_batch_size != current_batch_size:
                                    current_batch_size = new_batch_size
                                    dataloader = self._create_dataloader(current_batch_size)
                                    logger.info(f"重建 dataloader, batch_size={current_batch_size}")

                        # 日志
                        if self.global_step % self.config.logging_steps == 0:
                            logger.info(
                                f"  Step {self.global_step} | "
                                f"Loss: {loss:.4f} | "
                                f"显存: {memory_stats.get('current_gb', 0):.2f}GB | "
                                f"Batch: {current_batch_size} | "
                                f"时间: {step_time:.3f}s"
                            )

                except RuntimeError as e:
                    if "out of memory" in str(e):
                        self._handle_oom()
                        # 重建 dataloader
                        current_batch_size = self.batch_sizer.get_current_batch_size()
                        dataloader = self._create_dataloader(current_batch_size)
                        logger.info(f"OOM 后重建 dataloader, batch_size={current_batch_size}")
                        continue
                    else:
                        raise

            # Epoch 统计
            epoch_time = time.time() - epoch_start
            avg_loss = epoch_loss / max(epoch_steps, 1)
            logger.info(f"Epoch {epoch + 1} 完成: 平均 Loss={avg_loss:.4f}, 耗时={epoch_time:.1f}s")

        # 停止监控
        if self.config.enable_memory_aware:
            self.memory_monitor.stop()

        # 保存训练结果
        self._save_results()

        logger.info("=" * 60)
        logger.info("训练完成！")
        logger.info(f"总步数: {self.global_step}")
        logger.info(f"最终 Loss: {self.metrics_history[-1].loss:.4f}")
        logger.info("=" * 60)

        return self.metrics_history

    def _save_results(self):
        """保存训练结果"""
        os.makedirs(self.config.output_dir, exist_ok=True)

        # 保存指标
        results = {
            "config": {
                "model_name": self.config.model_name,
                "num_epochs": self.config.num_epochs,
                "learning_rate": self.config.learning_rate,
                "initial_batch_size": self.config.initial_batch_size,
                "enable_memory_aware": self.config.enable_memory_aware,
                "enable_smart_checkpoint": self.config.enable_smart_checkpoint,
                "enable_adaptive_batch": self.config.enable_adaptive_batch,
            },
            "summary": {
                "total_steps": self.global_step,
                "final_loss": self.metrics_history[-1].loss if self.metrics_history else 0,
                "min_loss": min(m.loss for m in self.metrics_history) if self.metrics_history else 0,
                "avg_step_time": sum(m.step_time for m in self.metrics_history) / max(len(self.metrics_history), 1),
                "peak_memory_gb": max(m.memory_allocated_gb for m in self.metrics_history) if self.metrics_history else 0,
            },
            "memory_monitor": self.memory_monitor.get_statistics(),
            "batch_sizer": self.batch_sizer.get_statistics(),
            "checkpoint_manager": self.checkpoint_manager.get_statistics(),
            "metrics": [m.to_dict() for m in self.metrics_history],
        }

        filepath = os.path.join(self.config.output_dir, "training_results.json")
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        logger.info(f"训练结果已保存: {filepath}")

    def get_summary(self) -> Dict[str, Any]:
        """获取训练摘要"""
        if not self.metrics_history:
            return {"status": "未开始"}

        return {
            "status": "已完成",
            "total_steps": self.global_step,
            "final_loss": self.metrics_history[-1].loss,
            "min_loss": min(m.loss for m in self.metrics_history),
            "avg_step_time": sum(m.step_time for m in self.metrics_history) / len(self.metrics_history),
            "peak_memory_gb": max(m.memory_allocated_gb for m in self.metrics_history),
            "batch_adjustments": self.batch_sizer.get_statistics()["total_adjustments"],
            "oom_count": self.batch_sizer.get_statistics()["oom_count"],
            "checkpoint_ratio": self.checkpoint_manager.get_checkpoint_ratio(),
        }
