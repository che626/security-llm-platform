"""
最小训练测试 - 验证 PyTorch + CUDA 能正常工作

用法：
    python training/test_minimal.py
"""

import torch
import time
import json
import os
from datetime import datetime


def get_gpu_info():
    """获取 GPU 信息"""
    if not torch.cuda.is_available():
        return {"available": False}

    props = torch.cuda.get_device_properties(0)
    return {
        "available": True,
        "name": props.name,
        "total_memory_gb": round(props.total_memory / 1024**3, 2),
        "cuda_version": torch.version.cuda,
        "pytorch_version": torch.__version__,
    }


def get_memory_usage():
    """获取当前显存使用"""
    if not torch.cuda.is_available():
        return {"allocated_gb": 0, "reserved_gb": 0, "usage_percent": 0}

    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3

    return {
        "allocated_gb": round(allocated, 3),
        "reserved_gb": round(reserved, 3),
        "total_gb": round(total, 2),
        "usage_percent": round(allocated / total * 100, 2),
    }


class SimpleModel(torch.nn.Module):
    """简单测试模型"""

    def __init__(self, vocab_size=32000, hidden_size=256, num_layers=4):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, hidden_size)
        self.layers = torch.nn.ModuleList([
            torch.nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=8,
                dim_feedforward=hidden_size * 4,
                batch_first=True,
            )
            for _ in range(num_layers)
        ])
        self.output = torch.nn.Linear(hidden_size, vocab_size)

    def forward(self, input_ids, labels=None, **kwargs):
        x = self.embedding(input_ids)
        for layer in self.layers:
            x = layer(x)
        logits = self.output(x)

        loss = None
        if labels is not None:
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
            )

        return type('Output', (), {'loss': loss, 'logits': logits})()


class DummyDataset(torch.utils.data.Dataset):
    """虚拟数据集"""

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


def run_training_test():
    """运行训练测试"""
    print("=" * 60)
    print("PyTorch + CUDA 训练测试")
    print("=" * 60)

    # 检查 GPU
    gpu_info = get_gpu_info()
    print(f"\nGPU 信息:")
    print(f"  可用: {gpu_info['available']}")
    if gpu_info['available']:
        print(f"  型号: {gpu_info['name']}")
        print(f"  显存: {gpu_info['total_memory_gb']} GB")
        print(f"  CUDA: {gpu_info['cuda_version']}")
        print(f"  PyTorch: {gpu_info['pytorch_version']}")

    # 创建模型
    print(f"\n创建模型...")
    model = SimpleModel(vocab_size=32000, hidden_size=256, num_layers=4)

    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  总参数量: {total_params:,}")
    print(f"  可训练参数: {trainable_params:,}")
    print(f"  模型大小: {total_params * 4 / 1024**2:.2f} MB (fp32)")

    # 移动到 GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"  设备: {device}")

    # 创建数据集
    dataset = DummyDataset(size=100, seq_length=128)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True)

    # 创建优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

    # 训练
    print(f"\n开始训练...")
    print(f"  Batch size: 2")
    print(f"  序列长度: 128")
    print(f"  训练步数: 20")

    model.train()
    step_times = []
    memory_history = []
    loss_history = []

    for step, batch in enumerate(dataloader):
        if step >= 20:
            break

        start_time = time.time()

        # 移动数据到 GPU
        batch = {k: v.to(device) for k, v in batch.items()}

        # 前向传播
        outputs = model(**batch)
        loss = outputs.loss

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 记录指标
        step_time = time.time() - start_time
        memory = get_memory_usage()

        step_times.append(step_time)
        memory_history.append(memory['allocated_gb'])
        loss_history.append(loss.item())

        # 打印进度
        if step % 5 == 0:
            print(f"  Step {step:3d} | Loss: {loss.item():.4f} | "
                  f"显存: {memory['allocated_gb']:.2f}GB | "
                  f"时间: {step_time:.3f}s")

    # 计算统计
    avg_step_time = sum(step_times) / len(step_times)
    peak_memory = max(memory_history)
    final_loss = loss_history[-1]
    throughput = 1 / avg_step_time

    print(f"\n训练完成！")
    print(f"  平均步时: {avg_step_time:.4f}s")
    print(f"  峰值显存: {peak_memory:.3f} GB")
    print(f"  最终 Loss: {final_loss:.4f}")
    print(f"  吞吐量: {throughput:.1f} steps/s")

    # 保存结果
    results = {
        "timestamp": datetime.now().isoformat(),
        "gpu_info": gpu_info,
        "model_config": {
            "vocab_size": 32000,
            "hidden_size": 256,
            "num_layers": 4,
            "total_params": total_params,
        },
        "training_config": {
            "batch_size": 2,
            "seq_length": 128,
            "learning_rate": 2e-5,
            "num_steps": 20,
        },
        "results": {
            "avg_step_time": round(avg_step_time, 4),
            "peak_memory_gb": round(peak_memory, 3),
            "final_loss": round(final_loss, 4),
            "throughput": round(throughput, 1),
        },
        "step_times": [round(t, 4) for t in step_times],
        "memory_history": [round(m, 3) for m in memory_history],
        "loss_history": [round(l, 4) for l in loss_history],
    }

    os.makedirs("outputs/benchmark", exist_ok=True)
    with open("outputs/benchmark/minimal_test.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n结果已保存: outputs/benchmark/minimal_test.json")

    return results


if __name__ == "__main__":
    run_training_test()
