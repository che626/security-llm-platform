"""
安全大模型训练脚本

使用 DeepSpeed ZeRO 进行安全大模型的微调训练。
支持 Qwen2.5、DeepSeek 等模型。

用法：
    # 单卡训练
    python train_security_llm.py --model qwen2.5-3b --zero_stage 2

    # 多卡训练
    deepspeed train_security_llm.py --model qwen2.5-3b --zero_stage 2 --num_gpus 2

    # 使用 LoRA
    python train_security_llm.py --model qwen2.5-3b --lora --zero_stage 2

    # 自动选择最优配置
    python train_security_llm.py --model qwen2.5-3b --adaptive
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
from torch.utils.data import Dataset, DataLoader

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class SecurityInstructionDataset(Dataset):
    """安全指令数据集"""

    def __init__(self, data_path: str, tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []

        # 加载数据
        if os.path.exists(data_path):
            import json
            with open(data_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        self.data.append(json.loads(line))
            logger.info(f"加载了 {len(self.data)} 条数据")
        else:
            logger.warning(f"数据文件不存在: {data_path}，使用示例数据")
            self.data = self._get_example_data()

    def _get_example_data(self):
        """获取示例数据"""
        return [
            {
                "instruction": "请分析以下 SSH 日志中的安全威胁",
                "input": "May 11 10:01:02 server sshd[1001]: Failed password for root from 8.8.8.8 port 53422 ssh2",
                "output": "该日志显示 SSH 暴力破解攻击。攻击者从 IP 8.8.8.8 尝试以 root 用户登录失败。建议：1) 封禁来源 IP 2) 检查是否有成功登录 3) 启用 Fail2Ban 4) 禁止 root 远程登录"
            },
            {
                "instruction": "请分析以下 Web 日志中的 SQL 注入攻击",
                "input": "1.2.3.4 - - \"GET /login.php?id=1' OR '1'='1 HTTP/1.1\" 200",
                "output": "该日志显示 SQL 注入攻击尝试。攻击者在 URL 参数中注入 ' OR '1'='1 绕过身份验证。建议：1) 使用参数化查询 2) 启用 WAF 规则 3) 审计数据库访问日志"
            },
            {
                "instruction": "请为 SSH 暴力破解事件生成 SOAR 响应剧本",
                "input": "检测到同一 IP 在 5 分钟内 SSH 登录失败超过 10 次",
                "output": "触发条件：同一 IP 5 分钟内 SSH 失败 >= 10 次\n响应动作：\n1. 自动封禁来源 IP 24 小时\n2. 通知安全管理员\n3. 收集主机认证日志\n4. 检查是否存在成功登录\n人工确认：封禁操作需要人工确认"
            },
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # 构建输入文本
        instruction = item.get("instruction", "")
        input_text = item.get("input", "")
        output_text = item.get("output", "")

        # 格式化为对话格式
        if input_text:
            prompt = f"### 指令：\n{instruction}\n\n### 输入：\n{input_text}\n\n### 回答：\n"
        else:
            prompt = f"### 指令：\n{instruction}\n\n### 回答：\n"

        full_text = prompt + output_text

        # Tokenize
        encodings = self.tokenizer(
            full_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        input_ids = encodings["input_ids"].squeeze()
        attention_mask = encodings["attention_mask"].squeeze()

        # 构建 labels（只计算输出部分的 loss）
        labels = input_ids.clone()

        # 将指令部分的 label 设为 -100（不计算 loss）
        prompt_tokens = self.tokenizer(
            prompt,
            max_length=self.max_length,
            truncation=True,
        )
        prompt_len = len(prompt_tokens["input_ids"])
        labels[:prompt_len] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="安全大模型训练脚本")

    # 模型参数
    parser.add_argument("--model", type=str, default="qwen2.5-3b",
                        help="模型名称或路径")
    parser.add_argument("--lora", action="store_true",
                        help="使用 LoRA 微调")
    parser.add_argument("--lora_rank", type=int, default=16,
                        help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32,
                        help="LoRA alpha")

    # 训练参数
    parser.add_argument("--epochs", type=int, default=3,
                        help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="每 GPU 批大小")
    parser.add_argument("--gradient_accumulation", type=int, default=4,
                        help="梯度累积步数")
    parser.add_argument("--lr", type=float, default=2e-5,
                        help="学习率")
    parser.add_argument("--max_seq_length", type=int, default=512,
                        help="最大序列长度")

    # DeepSpeed 参数
    parser.add_argument("--zero_stage", type=int, default=2, choices=[0, 1, 2, 3],
                        help="ZeRO Stage")
    parser.add_argument("--offload_optimizer", action="store_true",
                        help="Offload 优化器到 CPU")
    parser.add_argument("--offload_param", action="store_true",
                        help="Offload 参数到 CPU")
    parser.add_argument("--adaptive", action="store_true",
                        help="使用自适应配置")

    # 数据参数
    parser.add_argument("--data_path", type=str, default="data/security_instruction.jsonl",
                        help="训练数据路径")
    parser.add_argument("--output_dir", type=str, default="outputs/training",
                        help="输出目录")

    # 其他参数
    parser.add_argument("--num_gpus", type=int, default=1,
                        help="GPU 数量")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子")
    parser.add_argument("--logging_steps", type=int, default=10,
                        help="日志间隔")
    parser.add_argument("--save_steps", type=int, default=500,
                        help="保存间隔")

    return parser.parse_args()


def load_model(model_name: str, lora: bool = False, lora_rank: int = 16):
    """加载模型"""
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info(f"加载模型: {model_name}")

        # 加载 tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            padding_side="right",
        )

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # 加载模型
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )

        # 应用 LoRA
        if lora:
            try:
                from peft import LoraConfig, get_peft_model, TaskType

                lora_config = LoraConfig(
                    task_type=TaskType.CAUSAL_LM,
                    r=lora_rank,
                    lora_alpha=lora_rank * 2,
                    lora_dropout=0.1,
                    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
                )

                model = get_peft_model(model, lora_config)
                model.print_trainable_parameters()
                logger.info("LoRA 微调已启用")
            except ImportError:
                logger.warning("peft 未安装，使用全量微调")

        return model, tokenizer

    except Exception as e:
        logger.error(f"加载模型失败: {e}")
        raise


def main():
    """主函数"""
    args = parse_args()

    # 设置随机种子
    torch.manual_seed(args.seed)

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 加载模型
    model, tokenizer = load_model(
        args.model,
        lora=args.lora,
        lora_rank=args.lora_rank,
    )

    # 加载数据集
    dataset = SecurityInstructionDataset(
        data_path=args.data_path,
        tokenizer=tokenizer,
        max_length=args.max_seq_length,
    )

    # 创建训练器
    from src.deepspeed_optimizer import SecurityLLMTrainer, TrainingConfig

    config = TrainingConfig(
        model_name=args.model,
        output_dir=args.output_dir,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        max_seq_length=args.max_seq_length,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        zero_stage=-1 if args.adaptive else args.zero_stage,
        offload_optimizer=args.offload_optimizer,
        offload_param=args.offload_param,
        save_steps=args.save_steps,
    )

    trainer = SecurityLLMTrainer(
        model=model,
        train_dataset=dataset,
        config=config,
    )

    # 开始训练
    metrics = trainer.train()

    # 打印训练摘要
    summary = trainer.get_training_summary()
    logger.info("\n" + "=" * 60)
    logger.info("训练摘要:")
    for key, value in summary.items():
        if isinstance(value, dict):
            logger.info(f"  {key}:")
            for k, v in value.items():
                logger.info(f"    {k}: {v}")
        else:
            logger.info(f"  {key}: {value}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
