"""
ZeRO 基准测试运行脚本

用于对比原始 DeepSpeed ZeRO 与优化后的性能。

用法：
    # 运行默认基准测试
    python run_benchmark.py

    # 指定模型
    python run_benchmark.py --model qwen2.5-3b

    # 指定测试步数
    python run_benchmark.py --steps 200

    # 只测试特定配置
    python run_benchmark.py --configs baseline_zero2 optimized_zero2
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
from torch.utils.data import Dataset

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class DummyDataset(Dataset):
    """用于基准测试的虚拟数据集"""

    def __init__(self, size: int = 1000, seq_length: int = 512, vocab_size: int = 32000):
        self.size = size
        self.seq_length = seq_length
        self.vocab_size = vocab_size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return {
            "input_ids": torch.randint(0, self.vocab_size, (self.seq_length,)),
            "attention_mask": torch.ones(self.seq_length, dtype=torch.long),
            "labels": torch.randint(0, self.vocab_size, (self.seq_length,)),
        }


class DummyModel(torch.nn.Module):
    """用于基准测试的虚拟模型"""

    def __init__(self, vocab_size: int = 32000, hidden_size: int = 768, num_layers: int = 12):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, hidden_size)
        self.layers = torch.nn.ModuleList([
            torch.nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=12,
                dim_feedforward=hidden_size * 4,
                batch_first=True,
            )
            for _ in range(num_layers)
        ])
        self.output = torch.nn.Linear(hidden_size, vocab_size)

    def forward(self, input_ids, attention_mask=None, labels=None):
        x = self.embedding(input_ids)
        for layer in self.layers:
            x = layer(x)
        logits = self.output(x)

        loss = None
        if labels is not None:
            loss_fn = torch.nn.CrossEntropyLoss()
            loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))

        return type('Output', (), {'loss': loss, 'logits': logits})()


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="ZeRO 基准测试运行脚本")

    parser.add_argument("--model", type=str, default="dummy",
                        help="模型名称（dummy 表示使用虚拟模型）")
    parser.add_argument("--steps", type=int, default=100,
                        help="每个测试的步数")
    parser.add_argument("--output_dir", type=str, default="outputs/benchmark",
                        help="输出目录")
    parser.add_argument("--configs", nargs="+", default=None,
                        help="要测试的配置名称列表")
    parser.add_argument("--seq_length", type=int, default=512,
                        help="序列长度")
    parser.add_argument("--hidden_size", type=int, default=768,
                        help="隐藏层大小")
    parser.add_argument("--num_layers", type=int, default=12,
                        help="层数")

    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 定义模型和数据集工厂函数
    def model_factory():
        if args.model == "dummy":
            return DummyModel(
                vocab_size=32000,
                hidden_size=args.hidden_size,
                num_layers=args.num_layers,
            )
        else:
            from transformers import AutoModelForCausalLM
            return AutoModelForCausalLM.from_pretrained(
                args.model,
                torch_dtype=torch.float16,
                trust_remote_code=True,
            )

    def dataset_factory():
        return DummyDataset(
            size=1000,
            seq_length=args.seq_length,
            vocab_size=32000,
        )

    # 创建基准测试运行器
    from src.deepspeed_optimizer import ZeROBenchmark

    benchmark = ZeROBenchmark(
        model_factory=model_factory,
        dataset_factory=dataset_factory,
        output_dir=args.output_dir,
    )

    # 获取测试配置
    if args.configs:
        all_configs = benchmark.get_default_configs()
        configs = [c for c in all_configs if c.name in args.configs]
        if not configs:
            logger.error(f"未找到指定的配置: {args.configs}")
            logger.info(f"可用配置: {[c.name for c in all_configs]}")
            return
    else:
        configs = None  # 使用默认配置

    # 运行基准测试
    logger.info("=" * 60)
    logger.info("开始 ZeRO 基准测试")
    logger.info("=" * 60)

    results = benchmark.run_all(
        configs=configs,
        num_steps=args.steps,
    )

    # 生成对比报告
    report = benchmark.generate_comparison_report()
    report_path = os.path.join(args.output_dir, "benchmark_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    logger.info(f"对比报告已保存: {report_path}")

    # 生成图表数据
    charts = benchmark.generate_comparison_charts()
    import json
    charts_path = os.path.join(args.output_dir, "benchmark_charts.json")
    with open(charts_path, "w", encoding="utf-8") as f:
        json.dump(charts, f, indent=2, ensure_ascii=False)
    logger.info(f"图表数据已保存: {charts_path}")

    # 打印摘要
    logger.info("\n" + "=" * 60)
    logger.info("基准测试摘要")
    logger.info("=" * 60)

    for result in results:
        if result.success:
            logger.info(f"\n{result.config.name}:")
            logger.info(f"  描述: {result.config.description}")
            logger.info(f"  平均步时: {result.avg_step_time:.4f}s")
            logger.info(f"  吞吐量: {result.throughput_tokens_per_sec:.1f} tok/s")
            logger.info(f"  峰值显存: {result.peak_memory_gb:.2f} GB")
            logger.info(f"  最终 Loss: {result.final_loss:.4f}")
        else:
            logger.info(f"\n{result.config.name}: ❌ 失败 - {result.error_message}")

    logger.info("\n" + "=" * 60)
    logger.info(f"详细报告: {report_path}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
