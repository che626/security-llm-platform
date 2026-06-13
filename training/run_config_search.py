"""
自动配置搜索脚本

用法：
    python training/run_config_search.py
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset
import os
import sys

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.deepspeed_optimizer import AutoConfigSearch
from src.deepspeed_optimizer.config_search import ConfigSearchSpace


class SearchModel(nn.Module):
    """搜索测试模型"""

    def __init__(self, vocab_size=32000, hidden_size=256, num_layers=4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=8,
                dim_feedforward=hidden_size * 4,
                batch_first=True,
            )
            for _ in range(num_layers)
        ])
        self.output = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_ids, labels=None, **kwargs):
        x = self.embedding(input_ids)
        for layer in self.layers:
            x = layer(x)
        logits = self.output(x)

        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
            )

        return type('Output', (), {'loss': loss, 'logits': logits})()


class SearchDataset(Dataset):
    """搜索测试数据集"""

    def __init__(self, size=100, seq_length=128, vocab_size=32000):
        self.size = size
        self.seq_length = seq_length
        self.vocab_size = vocab_size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return {
            "input_ids": torch.randint(0, self.vocab_size, (self.seq_length,)),
            "labels": torch.randint(0, self.vocab_size, (self.seq_length,)),
        }


def main():
    """主函数"""
    print("=" * 60)
    print("自动配置搜索")
    print("=" * 60)

    # GPU 信息
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"\nGPU: {props.name}")
        print(f"显存: {props.total_memory / 1024**3:.2f} GB")

    # 模型工厂
    def model_factory():
        return SearchModel(vocab_size=32000, hidden_size=256, num_layers=4)

    # 数据集工厂
    def dataset_factory():
        return SearchDataset(size=100, seq_length=128)

    # 计算参数量
    model = model_factory()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params:,} ({total_params/1e6:.1f}M)")

    # 创建搜索器
    searcher = AutoConfigSearch(
        model_factory=model_factory,
        dataset_factory=dataset_factory,
        output_dir="outputs/search",
    )

    # 生成配置
    search_space = ConfigSearchSpace(
        stages=[0, 2, 3],
        batch_sizes=[2, 4, 8],
        gradient_accumulations=[1, 2],
    )
    configs = search_space.generate_default_configs()

    print(f"\n搜索配置数: {len(configs)}")
    print(f"每个配置测试步数: 20")

    # 执行搜索
    results = searcher.search(
        configs=configs,
        steps_per_trial=20,
        max_trials=10,
    )

    # 打印结果
    print("\n" + "=" * 60)
    print("搜索结果")
    print("=" * 60)

    print(f"\n总试验数: {results['total_trials']}")
    print(f"成功试验: {results['successful_trials']}")
    print(f"帕累托最优解: {len(results['pareto_optimal'])}")

    if results['pareto_optimal']:
        print("\n帕累托最优配置:")
        for i, config in enumerate(results['pareto_optimal'], 1):
            print(f"  {i}. {config['config']['name']}")
            print(f"     显存: {config['peak_memory_gb']:.2f} GB")
            print(f"     速度: {config['throughput']:.1f} tok/s")
            print(f"     Loss: {config['final_loss']:.4f}")

    # 打印报告
    print("\n" + "=" * 60)
    print("详细报告")
    print("=" * 60)
    # 替换 Unicode 字符避免编码错误
    report = results['report'].replace('✅', '[OK]').replace('❌', '[FAIL]')
    print(report)

    print(f"\n结果已保存: outputs/search/search_results.json")


if __name__ == "__main__":
    main()
