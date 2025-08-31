import os
import torch
import numpy as np
import evaluate
import nltk
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer

# --- 1. 配置 ---
model_id = "google/gemma-3-270m"
new_model_path = "./gemma-3-270m-teaching-eval"

# --- 2. 下载NLTK分词器数据 (评估指标需要) ---
try:
    nltk.data.find("tokenizers/punkt")
except (OSError, LookupError):
    print("NLTK 'punkt' aknizer not found. Downloading...")
    nltk.download("punkt", quiet=True)
    print("Download complete.")

# --- 3. 加载数据集 ---
print("Loading train and evaluation datasets...")
try:
    train_dataset = load_dataset('json', data_files='train_dataset.jsonl', split="train")
    eval_dataset = load_dataset('json', data_files='eval_dataset.jsonl', split="train")
except FileNotFoundError:
    print("错误: 未找到训练或验证数据集。")
    print("请先运行 split_dataset.py 来生成这些文件。")
    exit()

# --- 4. 加载模型和分词器 ---
print("Loading model and tokenizer...")
# 使用 4-bit 量化以节省显存 (如果GPU内存有限，请取消注释)
# bnb_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_compute_dtype=torch.bfloat16,
# )
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.float16,
    # quantization_config=bnb_config
)
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# --- 5. 定义评估指标计算函数 ---
print("Configuring metrics computation...")
rouge_metric = evaluate.load('rouge')
bleu_metric = evaluate.load('bleu')

def compute_metrics(eval_pred):
    predictions, labels = eval_pred

    # 将预测的token ID解码为文本
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)

    # 将标签中的-100替换为pad_token_id，然后解码
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # 为ROUGE和BLEU进行预处理：按句子分割
    decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
    decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]

    # 计算ROUGE分数
    rouge_result = rouge_metric.compute(predictions=decoded_preds, references=decoded_labels)

    # 计算BLEU分数
    bleu_result = bleu_metric.compute(
        predictions=[p.split() for p in decoded_preds],
        references=[[l.split()] for l in decoded_labels]
    )

    # 困惑度Perplexity会由Trainer自动计算并记录

    # 准备返回结果
    result = {
        "rouge2": rouge_result["rouge2"],
        "rougeL": rouge_result["rougeL"],
        "bleu": bleu_result["bleu"],
    }

    return {k: round(v, 4) for k, v in result.items()}

# --- 6. 配置PEFT (LoRA) ---
print("Configuring PEFT (LoRA)...")
lora_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=64,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
)
model = get_peft_model(model, lora_config)

# --- 7. 配置训练参数 ---
print("Setting training arguments...")
training_arguments = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,  # 增加训练轮次以便观察指标变化
    per_device_train_batch_size=2,
    gradient_accumulation_steps=2,
    optim="paged_adamw_32bit",

    # --- 新增评估和日志记录相关参数 ---
    evaluation_strategy="epoch",      # 每个epoch结束后进行评估
    save_strategy="epoch",            # 每个epoch结束后保存一次模型
    load_best_model_at_end=True,      # 训练结束后加载最优模型
    metric_for_best_model="eval_loss",# 使用验证集损失来判断最优模型
    predict_with_generate=True,       # 在评估时生成文本，以计算BLEU/ROUGE
    report_to="tensorboard",          # 将日志报告给TensorBoard
    # --- 参数结束 ---

    logging_steps=10,
    learning_rate=2e-4,
    weight_decay=0.001,
    fp16=True,
    max_grad_norm=0.3,
    warmup_ratio=0.03,
    group_by_length=True,
    lr_scheduler_type="cosine",
)

# --- 8. 初始化SFTTrainer ---
print("Initializing SFTTrainer...")
trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,          # 传入验证集
    peft_config=lora_config,
    dataset_text_field="text",
    max_seq_length=512,
    tokenizer=tokenizer,
    args=training_arguments,
    packing=False,
    compute_metrics=compute_metrics,    # 传入指标计算函数
)

# --- 9. 开始训练 ---
print("Starting training...")
trainer.train()
print("Training finished.")

# --- 10. 保存最终模型 ---
print(f"Saving final model to {new_model_path}...")
trainer.model.save_pretrained(new_model_path)
print("Model saved successfully.")

print("\nFine-tuning process complete.")
print(f"To view training metrics, run: tensorboard --logdir={training_arguments.output_dir}/runs")
