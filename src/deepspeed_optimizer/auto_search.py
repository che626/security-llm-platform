"""
自动配置搜索器

核心创新：
自动测试不同配置组合，找到帕累托最优解（速度 vs 显存）。
用户无需手动试错，系统自动推荐最优配置。
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import time
import json
import os
import logging
from typing import List, Dict, Any, Optional, Callable
from datetime import datetime

from .config_search import SearchConfig, ConfigSearchSpace
from .memory_monitor import MemoryMonitor
from .adaptive_batch import AdaptiveBatchSizer
from .smart_checkpoint import SmartCheckpointManager

logger = logging.getLogger(__name__)


class SearchResult:
    """搜索结果"""

    def __init__(self, config: SearchConfig):
        self.config = config
        self.success = False
        self.error_message = ""

        # 性能指标
        self.avg_step_time = 0.0
        self.peak_memory_gb = 0.0
        self.throughput = 0.0
        self.final_loss = 0.0

        # 详细数据
        self.step_times: List[float] = []
        self.memory_history: List[float] = []
        self.loss_history: List[float] = []

    def to_dict(self) -> Dict[str, Any]:
        return {
            "config": {
                "name": self.config.name,
                "stage": self.config.stage,
                "offload_optimizer": self.config.offload_optimizer,
                "batch_size": self.config.batch_size,
                "gradient_accumulation": self.config.gradient_accumulation,
                "use_smart_checkpoint": self.config.use_smart_checkpoint,
                "use_adaptive_batch": self.config.use_adaptive_batch,
            },
            "success": self.success,
            "error_message": self.error_message,
            "avg_step_time": round(self.avg_step_time, 4),
            "peak_memory_gb": round(self.peak_memory_gb, 3),
            "throughput": round(self.throughput, 2),
            "final_loss": round(self.final_loss, 6),
        }


class AutoConfigSearch:
    """
    自动配置搜索器

    功能：
    1. 自动测试多种配置组合
    2. 收集性能指标（速度、显存、Loss）
    3. 找到帕累托最优解
    4. 生成对比报告
    """

    def __init__(
        self,
        model_factory: Callable[[], nn.Module],
        dataset_factory: Callable[[], Dataset],
        output_dir: str = "outputs/search",
    ):
        self.model_factory = model_factory
        self.dataset_factory = dataset_factory
        self.output_dir = output_dir

        os.makedirs(output_dir, exist_ok=True)

        logger.info("AutoConfigSearch 初始化")

    def search(
        self,
        configs: Optional[List[SearchConfig]] = None,
        steps_per_trial: int = 30,
        max_trials: int = 20,
    ) -> Dict[str, Any]:
        """
        执行配置搜索

        Args:
            configs: 配置列表，None 使用默认配置
            steps_per_trial: 每次试验的训练步数
            max_trials: 最大试验次数

        Returns:
            搜索结果
        """
        if configs is None:
            search_space = ConfigSearchSpace()
            configs = search_space.generate_default_configs()

        # 限制试验次数
        if len(configs) > max_trials:
            import random
            configs = random.sample(configs, max_trials)

        logger.info(f"开始配置搜索: {len(configs)} 个配置, 每个 {steps_per_trial} 步")

        results: List[SearchResult] = []

        for i, config in enumerate(configs):
            logger.info(f"\n试验 {i+1}/{len(configs)}: {config.name}")

            result = self._run_trial(config, steps_per_trial)
            results.append(result)

            if result.success:
                logger.info(
                    f"  结果: 显存={result.peak_memory_gb:.2f}GB, "
                    f"速度={result.throughput:.1f}tok/s, "
                    f"Loss={result.final_loss:.4f}"
                )
            else:
                logger.info(f"  失败: {result.error_message}")

        # 找到帕累托最优解
        successful_results = [r for r in results if r.success]
        pareto_optimal = self._find_pareto_optimal(successful_results)

        # 生成报告
        report = self._generate_report(results, pareto_optimal)

        # 保存结果
        self._save_results(results, pareto_optimal)

        return {
            "total_trials": len(configs),
            "successful_trials": len(successful_results),
            "pareto_optimal": [r.to_dict() for r in pareto_optimal],
            "all_results": [r.to_dict() for r in results],
            "report": report,
        }

    def _run_trial(self, config: SearchConfig, num_steps: int) -> SearchResult:
        """运行单次试验"""
        result = SearchResult(config)

        try:
            # 创建模型和数据集
            model = self.model_factory()
            dataset = self.dataset_factory()

            # 设备
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = model.to(device)

            # 优化器
            optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

            # 内存监控
            memory_monitor = MemoryMonitor()
            memory_monitor.start()

            # 自适应批大小
            if config.use_adaptive_batch:
                batch_sizer = AdaptiveBatchSizer(initial_batch_size=config.batch_size)
                current_batch_size = batch_sizer.get_current_batch_size()
            else:
                current_batch_size = config.batch_size

            # 智能检查点
            if config.use_smart_checkpoint:
                checkpoint_manager = SmartCheckpointManager()
                memory_usage = memory_monitor.get_current_usage()
                checkpoint_manager.apply_smart_checkpointing(model, memory_usage)

            # 数据加载器
            dataloader = DataLoader(dataset, batch_size=current_batch_size, shuffle=True)

            # 训练
            model.train()
            step_times = []
            memory_history = []
            loss_history = []

            for step, batch in enumerate(dataloader):
                if step >= num_steps:
                    break

                start_time = time.time()

                try:
                    # 移动数据
                    batch = {k: v.to(device) for k, v in batch.items()}

                    # 前向传播
                    outputs = model(**batch)
                    loss = outputs.loss if hasattr(outputs, 'loss') else outputs[0]

                    # 反向传播
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    # 记录
                    step_time = time.time() - start_time
                    memory = torch.cuda.memory_allocated() / 1024**3

                    step_times.append(step_time)
                    memory_history.append(memory)
                    loss_history.append(loss.item())

                    # 自适应调整
                    if config.use_adaptive_batch:
                        memory_usage = memory_monitor.get_current_usage()
                        new_batch_size = batch_sizer.adjust(memory_usage)
                        if new_batch_size != current_batch_size:
                            current_batch_size = new_batch_size
                            dataloader = DataLoader(dataset, batch_size=current_batch_size, shuffle=True)

                except RuntimeError as e:
                    if "out of memory" in str(e):
                        logger.warning(f"  Step {step}: OOM")
                        torch.cuda.empty_cache()

                        if config.use_adaptive_batch:
                            batch_sizer.adjust(1.0, oom_occurred=True)
                            current_batch_size = batch_sizer.get_current_batch_size()
                            dataloader = DataLoader(dataset, batch_size=current_batch_size, shuffle=True)
                        else:
                            result.error_message = "OOM"
                            break
                    else:
                        raise

            # 停止监控
            memory_monitor.stop()

            # 计算结果
            if step_times:
                result.success = True
                result.avg_step_time = sum(step_times) / len(step_times)
                result.peak_memory_gb = max(memory_history) if memory_history else 0
                result.throughput = 1 / result.avg_step_time if result.avg_step_time > 0 else 0
                result.final_loss = loss_history[-1] if loss_history else 0
                result.step_times = step_times
                result.memory_history = memory_history
                result.loss_history = loss_history

        except Exception as e:
            result.error_message = str(e)
            logger.error(f"  试验失败: {e}")

        return result

    def _find_pareto_optimal(self, results: List[SearchResult]) -> List[SearchResult]:
        """
        找到帕累托最优解

        帕累托最优：没有其他配置同时在速度和显存两个维度上都优于它。
        """
        if not results:
            return []

        pareto = []

        for r in results:
            is_dominated = False

            for other in results:
                if other is r:
                    continue

                # 检查是否被支配
                # other 优于 r 的条件：速度更快且显存更低
                if (other.throughput >= r.throughput and
                    other.peak_memory_gb <= r.peak_memory_gb and
                    (other.throughput > r.throughput or
                     other.peak_memory_gb < r.peak_memory_gb)):
                    is_dominated = True
                    break

            if not is_dominated:
                pareto.append(r)

        # 按速度排序
        pareto.sort(key=lambda x: x.throughput, reverse=True)

        return pareto

    def _generate_report(
        self,
        all_results: List[SearchResult],
        pareto_optimal: List[SearchResult],
    ) -> str:
        """生成搜索报告"""
        lines = [
            "# 自动配置搜索报告",
            "",
            f"**搜索时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"**总试验数**: {len(all_results)}",
            f"**成功试验**: {sum(1 for r in all_results if r.success)}",
            f"**帕累托最优解**: {len(pareto_optimal)}",
            "",
            "## 帕累托最优配置",
            "",
            "| 配置 | Stage | Batch | 显存 (GB) | 速度 (tok/s) | Loss |",
            "|------|-------|-------|-----------|--------------|------|",
        ]

        for r in pareto_optimal:
            lines.append(
                f"| {r.config.name} | {r.config.stage} | {r.config.batch_size} | "
                f"{r.peak_memory_gb:.2f} | {r.throughput:.1f} | {r.final_loss:.4f} |"
            )

        lines.extend([
            "",
            "## 所有结果",
            "",
            "| 配置 | Stage | Batch | 显存 (GB) | 速度 (tok/s) | Loss | 状态 |",
            "|------|-------|-------|-----------|--------------|------|------|",
        ])

        for r in all_results:
            status = "✅" if r.success else "❌"
            if r.success:
                lines.append(
                    f"| {r.config.name} | {r.config.stage} | {r.config.batch_size} | "
                    f"{r.peak_memory_gb:.2f} | {r.throughput:.1f} | {r.final_loss:.4f} | {status} |"
                )
            else:
                lines.append(
                    f"| {r.config.name} | {r.config.stage} | {r.config.batch_size} | "
                    f"- | - | - | {status} |"
                )

        lines.extend([
            "",
            "## 推荐",
            "",
        ])

        if pareto_optimal:
            best = pareto_optimal[0]
            lines.append(f"**最佳配置**: {best.config.name}")
            lines.append(f"- Stage: {best.config.stage}")
            lines.append(f"- Batch Size: {best.config.batch_size}")
            lines.append(f"- 智能检查点: {'启用' if best.config.use_smart_checkpoint else '禁用'}")
            lines.append(f"- 自适应批大小: {'启用' if best.config.use_adaptive_batch else '禁用'}")
            lines.append(f"- 预期速度: {best.throughput:.1f} tok/s")
            lines.append(f"- 预期显存: {best.peak_memory_gb:.2f} GB")

        return "\n".join(lines)

    def _save_results(
        self,
        all_results: List[SearchResult],
        pareto_optimal: List[SearchResult],
    ):
        """保存搜索结果"""
        results_data = {
            "timestamp": datetime.now().isoformat(),
            "total_trials": len(all_results),
            "successful_trials": sum(1 for r in all_results if r.success),
            "pareto_optimal": [r.to_dict() for r in pareto_optimal],
            "all_results": [r.to_dict() for r in all_results],
        }

        filepath = os.path.join(self.output_dir, "search_results.json")
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(results_data, f, indent=2, ensure_ascii=False)

        logger.info(f"搜索结果已保存: {filepath}")
