"""
配置搜索空间定义

定义所有可搜索的配置参数及其取值范围。
"""

from typing import List, Dict, Any
from dataclasses import dataclass


@dataclass
class SearchConfig:
    """单个搜索配置"""
    name: str
    stage: int
    offload_optimizer: bool
    offload_param: bool
    batch_size: int
    gradient_accumulation: int
    fp16: bool
    use_smart_checkpoint: bool
    use_adaptive_batch: bool


class ConfigSearchSpace:
    """
    配置搜索空间

    定义所有可搜索的配置参数。
    """

    def __init__(
        self,
        stages: List[int] = None,
        offload_options: List[bool] = None,
        batch_sizes: List[int] = None,
        gradient_accumulations: List[int] = None,
    ):
        self.stages = stages or [0, 1, 2, 3]
        self.offload_options = offload_options or [True, False]
        self.batch_sizes = batch_sizes or [1, 2, 4, 8]
        self.gradient_accumulations = gradient_accumulations or [1, 2, 4]

    def generate_configs(self) -> List[SearchConfig]:
        """生成所有配置组合"""
        configs = []
        config_id = 0

        for stage in self.stages:
            for offload in self.offload_options:
                for batch_size in self.batch_sizes:
                    for grad_accum in self.gradient_accumulations:
                        # 跳过无效组合
                        if stage < 2 and offload:
                            continue  # ZeRO-0/1 不支持 offload

                        config_id += 1

                        # 基础配置（无优化）
                        configs.append(SearchConfig(
                            name=f"baseline_s{stage}_b{batch_size}_ga{grad_accum}",
                            stage=stage,
                            offload_optimizer=offload,
                            offload_param=False,
                            batch_size=batch_size,
                            gradient_accumulation=grad_accum,
                            fp16=True,
                            use_smart_checkpoint=False,
                            use_adaptive_batch=False,
                        ))

                        # 优化配置（使用我们的创新）
                        configs.append(SearchConfig(
                            name=f"optimized_s{stage}_b{batch_size}_ga{grad_accum}",
                            stage=stage,
                            offload_optimizer=offload,
                            offload_param=False,
                            batch_size=batch_size,
                            gradient_accumulation=grad_accum,
                            fp16=True,
                            use_smart_checkpoint=True,
                            use_adaptive_batch=True,
                        ))

        return configs

    def generate_default_configs(self) -> List[SearchConfig]:
        """生成默认测试配置（用于快速测试）"""
        return [
            # 基准配置
            SearchConfig(
                name="baseline_s0_b4",
                stage=0,
                offload_optimizer=False,
                offload_param=False,
                batch_size=4,
                gradient_accumulation=1,
                fp16=True,
                use_smart_checkpoint=False,
                use_adaptive_batch=False,
            ),
            SearchConfig(
                name="baseline_s2_b4",
                stage=2,
                offload_optimizer=False,
                offload_param=False,
                batch_size=4,
                gradient_accumulation=1,
                fp16=True,
                use_smart_checkpoint=False,
                use_adaptive_batch=False,
            ),
            SearchConfig(
                name="baseline_s3_b4",
                stage=3,
                offload_optimizer=False,
                offload_param=False,
                batch_size=4,
                gradient_accumulation=1,
                fp16=True,
                use_smart_checkpoint=False,
                use_adaptive_batch=False,
            ),

            # 优化配置
            SearchConfig(
                name="optimized_s0_b4",
                stage=0,
                offload_optimizer=False,
                offload_param=False,
                batch_size=4,
                gradient_accumulation=1,
                fp16=True,
                use_smart_checkpoint=True,
                use_adaptive_batch=True,
            ),
            SearchConfig(
                name="optimized_s2_b4",
                stage=2,
                offload_optimizer=False,
                offload_param=False,
                batch_size=4,
                gradient_accumulation=1,
                fp16=True,
                use_smart_checkpoint=True,
                use_adaptive_batch=True,
            ),
            SearchConfig(
                name="optimized_s3_b4",
                stage=3,
                offload_optimizer=False,
                offload_param=False,
                batch_size=4,
                gradient_accumulation=1,
                fp16=True,
                use_smart_checkpoint=True,
                use_adaptive_batch=True,
            ),
        ]
