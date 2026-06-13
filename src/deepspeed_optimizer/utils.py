"""
DeepSpeed 优化器工具函数
"""

import os
import json
import logging
from typing import Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


def get_gpu_info() -> Dict[str, Any]:
    """获取 GPU 信息"""
    try:
        import torch
        if not torch.cuda.is_available():
            return {"available": False}

        gpus = []
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            gpus.append({
                "id": i,
                "name": props.name,
                "total_memory_gb": round(props.total_mem / (1024 ** 3), 2),
                "major": props.major,
                "minor": props.minor,
                "multi_processor_count": props.multi_processor_count,
            })

        return {
            "available": True,
            "count": len(gpus),
            "gpus": gpus,
            "cuda_version": torch.version.cuda,
            "pytorch_version": torch.__version__,
        }
    except Exception as e:
        return {"available": False, "error": str(e)}


def get_model_info(model) -> Dict[str, Any]:
    """获取模型信息"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "total_params_billions": round(total_params / 1e9, 3),
        "trainable_params_billions": round(trainable_params / 1e9, 3),
        "model_size_gb": round(total_params * 4 / (1024 ** 3), 2),  # fp32
        "trainable_ratio": round(trainable_params / total_params * 100, 2) if total_params > 0 else 0,
    }


def save_training_config(config: Dict[str, Any], output_dir: str, filename: str = "training_config.json"):
    """保存训练配置"""
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    logger.info(f"训练配置已保存: {filepath}")
    return filepath


def load_training_config(config_path: str) -> Dict[str, Any]:
    """加载训练配置"""
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    logger.info(f"训练配置已加载: {config_path}")
    return config


def format_bytes(bytes_value: int) -> str:
    """格式化字节数"""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if bytes_value < 1024.0:
            return f"{bytes_value:.2f} {unit}"
        bytes_value /= 1024.0
    return f"{bytes_value:.2f} PB"


def format_time(seconds: float) -> str:
    """格式化时间"""
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.2f}min"
    else:
        hours = seconds / 3600
        return f"{hours:.2f}h"


def calculate_speedup(baseline_time: float, optimized_time: float) -> float:
    """计算加速比"""
    if optimized_time == 0:
        return float("inf")
    return baseline_time / optimized_time


def calculate_memory_savings(baseline_memory: float, optimized_memory: float) -> float:
    """计算显存节省比例"""
    if baseline_memory == 0:
        return 0.0
    return (baseline_memory - optimized_memory) / baseline_memory * 100


def ensure_directory(path: str) -> str:
    """确保目录存在"""
    os.makedirs(path, exist_ok=True)
    return path


def list_checkpoints(output_dir: str) -> list:
    """列出所有检查点"""
    checkpoints = []
    if os.path.exists(output_dir):
        for item in os.listdir(output_dir):
            if item.startswith("checkpoint-") or item == "final":
                checkpoint_path = os.path.join(output_dir, item)
                if os.path.isdir(checkpoint_path):
                    checkpoints.append({
                        "name": item,
                        "path": checkpoint_path,
                        "step": int(item.split("-")[-1]) if "-" in item else float("inf"),
                    })

    checkpoints.sort(key=lambda x: x["step"])
    return checkpoints


def get_latest_checkpoint(output_dir: str) -> Optional[str]:
    """获取最新的检查点"""
    checkpoints = list_checkpoints(output_dir)
    if checkpoints:
        return checkpoints[-1]["path"]
    return None
