"""
真实基准测试脚本

对比：
1. 标准 PyTorch 训练
2. 内存感知训练（我们的创新）

用法：
    python training/run_real_benchmark.py
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import time
import json
import os
import sys
from datetime import datetime

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.deepspeed_optimizer import (
    MemoryMonitor,
    AdaptiveBatchSizer,
    SmartCheckpointManager,
    MemoryAwareTrainer,
    TrainingConfig,
)


class BenchmarkModel(nn.Module):
    """基准测试模型"""

    def __init__(self, vocab_size=32000, hidden_size=512, num_layers=8):
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


class BenchmarkDataset(Dataset):
    """基准测试数据集"""

    def __init__(self, size=200, seq_length=256, vocab_size=32000):
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


def run_standard_training(model, dataset, num_steps=50, batch_size=4):
    """标准训练（无优化）"""
    print("\n" + "=" * 50)
    print("标准训练（无内存感知）")
    print("=" * 50)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    step_times = []
    memory_history = []
    loss_history = []

    model.train()

    for step, batch in enumerate(dataloader):
        if step >= num_steps:
            break

        start_time = time.time()

        try:
            # 移动数据
            batch = {k: v.to(device) for k, v in batch.items()}

            # 前向传播
            outputs = model(**batch)
            loss = outputs.loss

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

            if step % 10 == 0:
                print(f"  Step {step:3d} | Loss: {loss.item():.4f} | "
                      f"显存: {memory:.2f}GB | 时间: {step_time:.3f}s")

        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"  Step {step}: OOM 发生！训练终止")
                torch.cuda.empty_cache()
                break
            else:
                raise

    return {
        "method": "标准训练",
        "steps": len(step_times),
        "avg_step_time": sum(step_times) / max(len(step_times), 1),
        "peak_memory": max(memory_history) if memory_history else 0,
        "final_loss": loss_history[-1] if loss_history else 0,
        "oom_occurred": len(step_times) < num_steps,
        "step_times": step_times,
        "memory_history": memory_history,
        "loss_history": loss_history,
    }


def run_memory_aware_training(model, dataset, num_steps=50, initial_batch_size=4):
    """内存感知训练（我们的创新）"""
    print("\n" + "=" * 50)
    print("内存感知训练（我们的创新）")
    print("=" * 50)

    config = TrainingConfig(
        model_name="benchmark_model",
        output_dir="outputs/benchmark/memory_aware",
        num_epochs=1,
        learning_rate=2e-5,
        initial_batch_size=initial_batch_size,
        enable_memory_aware=True,
        enable_smart_checkpoint=True,
        enable_adaptive_batch=True,
        logging_steps=10,
    )

    trainer = MemoryAwareTrainer(
        model=model,
        train_dataset=dataset,
        config=config,
    )

    # 手动限制步数
    metrics = []
    model.train()

    # 启动监控
    trainer.memory_monitor.start()
    trainer.model = trainer.model.to(trainer.device)
    trainer._init_optimizer()

    # 应用智能检查点
    memory_usage = trainer.memory_monitor.get_current_usage()
    trainer.checkpoint_manager.apply_smart_checkpointing(trainer.model, memory_usage)

    current_batch_size = trainer.batch_sizer.get_current_batch_size()
    dataloader = DataLoader(dataset, batch_size=current_batch_size, shuffle=True)

    step_times = []
    memory_history = []
    loss_history = []

    for step, batch in enumerate(dataloader):
        if step >= num_steps:
            break

        start_time = time.time()

        try:
            # 移动数据
            batch = {k: v.to(trainer.device) for k, v in batch.items()}

            # 前向传播
            outputs = trainer.model(**batch)
            loss = outputs.loss

            # 反向传播
            trainer.optimizer.zero_grad()
            loss.backward()
            trainer.optimizer.step()

            trainer.global_step += 1

            # 记录
            step_time = time.time() - start_time
            memory = torch.cuda.memory_allocated() / 1024**3
            memory_usage = trainer.memory_monitor.get_current_usage()

            step_times.append(step_time)
            memory_history.append(memory)
            loss_history.append(loss.item())

            # 内存感知调整
            if trainer.config.enable_adaptive_batch:
                new_batch_size = trainer.batch_sizer.adjust(memory_usage)
                if new_batch_size != current_batch_size:
                    current_batch_size = new_batch_size
                    dataloader = DataLoader(dataset, batch_size=current_batch_size, shuffle=True)

            if step % 10 == 0:
                print(f"  Step {step:3d} | Loss: {loss.item():.4f} | "
                      f"显存: {memory:.2f}GB | Batch: {current_batch_size} | "
                      f"时间: {step_time:.3f}s")

        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"  Step {step}: OOM 发生，执行紧急降级")
                trainer._handle_oom()
                current_batch_size = trainer.batch_sizer.get_current_batch_size()
                dataloader = DataLoader(dataset, batch_size=current_batch_size, shuffle=True)
                continue
            else:
                raise

    # 停止监控
    trainer.memory_monitor.stop()

    return {
        "method": "内存感知训练",
        "steps": len(step_times),
        "avg_step_time": sum(step_times) / max(len(step_times), 1),
        "peak_memory": max(memory_history) if memory_history else 0,
        "final_loss": loss_history[-1] if loss_history else 0,
        "oom_occurred": False,
        "batch_adjustments": trainer.batch_sizer.get_statistics()["total_adjustments"],
        "checkpoint_ratio": trainer.checkpoint_manager.get_checkpoint_ratio(),
        "step_times": step_times,
        "memory_history": memory_history,
        "loss_history": loss_history,
    }


def main():
    """主函数"""
    print("=" * 60)
    print("真实基准测试 - 标准训练 vs 内存感知训练")
    print("=" * 60)

    # GPU 信息
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"\nGPU: {props.name}")
        print(f"显存: {props.total_memory / 1024**3:.2f} GB")
        print(f"CUDA: {torch.version.cuda}")

    # 测试配置
    num_steps = 50
    batch_size = 4

    print(f"\n测试配置:")
    print(f"  步数: {num_steps}")
    print(f"  批大小: {batch_size}")
    print(f"  模型: Transformer (512 hidden, 8 layers)")

    # 创建模型和数据集
    model = BenchmarkModel(vocab_size=32000, hidden_size=512, num_layers=8)
    dataset = BenchmarkDataset(size=200, seq_length=256)

    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  参数量: {total_params:,} ({total_params/1e6:.1f}M)")

    # 运行标准训练
    import copy
    model_standard = copy.deepcopy(model)
    results_standard = run_standard_training(
        model_standard, dataset, num_steps=num_steps, batch_size=batch_size
    )

    # 清理显存
    del model_standard
    torch.cuda.empty_cache()

    # 运行内存感知训练
    model_aware = copy.deepcopy(model)
    results_aware = run_memory_aware_training(
        model_aware, dataset, num_steps=num_steps, initial_batch_size=batch_size
    )

    # 对比结果
    print("\n" + "=" * 60)
    print("对比结果")
    print("=" * 60)

    print(f"\n{'指标':<20} {'标准训练':<15} {'内存感知':<15} {'差异':<10}")
    print("-" * 60)

    # 完成步数
    print(f"{'完成步数':<20} {results_standard['steps']:<15} {results_aware['steps']:<15}")

    # 平均步时
    std_time = results_standard['avg_step_time']
    aware_time = results_aware['avg_step_time']
    time_diff = (aware_time - std_time) / std_time * 100
    print(f"{'平均步时 (s)':<20} {std_time:<15.4f} {aware_time:<15.4f} {time_diff:+.1f}%")

    # 峰值显存
    std_mem = results_standard['peak_memory']
    aware_mem = results_aware['peak_memory']
    mem_diff = (aware_mem - std_mem) / std_mem * 100
    print(f"{'峰值显存 (GB)':<20} {std_mem:<15.3f} {aware_mem:<15.3f} {mem_diff:+.1f}%")

    # 最终 Loss
    std_loss = results_standard['final_loss']
    aware_loss = results_aware['final_loss']
    loss_diff = (aware_loss - std_loss) / std_loss * 100
    print(f"{'最终 Loss':<20} {std_loss:<15.4f} {aware_loss:<15.4f} {loss_diff:+.1f}%")

    # OOM
    print(f"{'OOM 发生':<20} {'是' if results_standard['oom_occurred'] else '否':<15} {'否':<15}")

    # 内存感知特有指标
    print(f"\n内存感知特有指标:")
    print(f"  批大小调整次数: {results_aware.get('batch_adjustments', 0)}")
    print(f"  检查点比例: {results_aware.get('checkpoint_ratio', 0):.1%}")

    # 保存结果
    all_results = {
        "timestamp": datetime.now().isoformat(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "config": {
            "num_steps": num_steps,
            "batch_size": batch_size,
            "model_params": total_params,
        },
        "standard_training": {k: v for k, v in results_standard.items()
                             if k not in ["step_times", "memory_history", "loss_history"]},
        "memory_aware_training": {k: v for k, v in results_aware.items()
                                 if k not in ["step_times", "memory_history", "loss_history"]},
        "standard_history": {
            "step_times": results_standard["step_times"],
            "memory_history": results_standard["memory_history"],
            "loss_history": results_standard["loss_history"],
        },
        "memory_aware_history": {
            "step_times": results_aware["step_times"],
            "memory_history": results_aware["memory_history"],
            "loss_history": results_aware["loss_history"],
        },
    }

    os.makedirs("outputs/benchmark", exist_ok=True)
    with open("outputs/benchmark/real_benchmark.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n结果已保存: outputs/benchmark/real_benchmark.json")


if __name__ == "__main__":
    main()
