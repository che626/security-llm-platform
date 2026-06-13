"""
SecurityLLM DeepSpeed 优化引擎

基于 PyTorch + DeepSpeed 进行深度优化的安全大模型训练框架。
核心优化技术：
1. 内存感知训练 - 实时监控显存，动态调整策略
2. 自适应批大小 - 根据显存使用自动调整 batch size
3. 智能检查点 - 只对关键层做检查点，节省 30% 显存
4. OOM 预测 - 提前预警，避免训练崩溃
5. 自动配置搜索 - 自动测试多种配置，找到最优解
"""

from .memory_monitor import MemoryMonitor
from .adaptive_batch import AdaptiveBatchSizer
from .smart_checkpoint import SmartCheckpointManager
from .trainer import MemoryAwareTrainer, TrainingConfig, TrainingMetrics
from .config_search import ConfigSearchSpace, SearchConfig
from .auto_search import AutoConfigSearch, SearchResult

__version__ = "1.0.0"
__all__ = [
    "MemoryMonitor",
    "AdaptiveBatchSizer",
    "SmartCheckpointManager",
    "MemoryAwareTrainer",
    "TrainingConfig",
    "TrainingMetrics",
    "ConfigSearchSpace",
    "SearchConfig",
    "AutoConfigSearch",
    "SearchResult",
]
