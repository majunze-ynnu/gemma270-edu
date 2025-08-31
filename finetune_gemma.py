import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline,
)
from peft import LoraConfig, PeftModel, get_peft_model
from trl import SFTTrainer

# 1. 配置模型和分词器 (Model and Tokenizer Configuration)
# ---
# 模型ID，我们选择 Google 的 gemma-3-270m
model_id = "google/gemma-3-270m"
# 数据集文件路径
dataset_path = "teaching_eval_dataset.jsonl"
# 微调后模型的保存路径
new_model_path = "./gemma-3-270m-teaching-eval"

# 2. 加载数据集 (Load Dataset)
# ---
# 使用 datasets 库加载 JSONL 文件
print("Loading dataset...")
dataset = load_dataset('json', data_files=dataset_path, split="train")

# 3. 配置量化 (Quantization Configuration)
# ---
# 使用 4-bit 量化以节省显存
# bnb_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_compute_dtype=torch.bfloat16,
#     bnb_4bit_use_double_quant=False,
# )
# Note: To run on a T4 GPU (like in Colab), you might need to use the above config.
# For wider compatibility, we'll load in float16 for now. If you get OOM errors,
# uncomment the BitsAndBytesConfig and add `quantization_config=bnb_config` to `from_pretrained`.
print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.float16 # Use float16 to save memory
)

# 4. 加载分词器 (Load Tokenizer)
# ---
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
# Gemma-2b 的 pad_token 可能未设置，我们将其设置为 eos_token
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right" # 右侧填充，以避免问题

# 5. 配置 PEFT (LoRA)
# ---
print("Configuring PEFT (LoRA)...")
# LoRA 配置
lora_config = LoraConfig(
    lora_alpha=16,          # LoRA 缩放因子
    lora_dropout=0.1,       # Dropout 率
    r=64,                   # LoRA 的秩
    bias="none",            # 不训练偏置项
    task_type="CAUSAL_LM",
    # 针对 Gemma 模型的特定模块
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
)

# 将 PEFT 应用于模型
model = get_peft_model(model, lora_config)
print("PEFT model created.")

# 6. 配置训练参数 (Training Arguments)
# ---
print("Setting training arguments...")
training_arguments = TrainingArguments(
    output_dir="./results",                     # 训练输出目录
    num_train_epochs=1,                         # 训练轮次
    per_device_train_batch_size=2,              # 每个设备的批大小
    gradient_accumulation_steps=2,              # 梯度累积步数，有效批大小为 2*2=4
    optim="paged_adamw_32bit",                  # 使用分页 AdamW 优化器以节省内存
    save_steps=50,                              # 每 50 步保存一次检查点
    logging_steps=10,                           # 每 10 步记录一次日志
    learning_rate=2e-4,                         # 学习率
    weight_decay=0.001,                         # 权重衰减
    fp16=True,                                  # 使用 16 位浮点数训练
    bf16=False,                                 # 不使用 bfloat16
    max_grad_norm=0.3,                          # 最大梯度范数
    max_steps=-1,                               # 如果设置为正数，则覆盖 num_train_epochs
    warmup_ratio=0.03,                          # 预热比例
    group_by_length=True,                       # 按长度对样本进行分组，以提高效率
    lr_scheduler_type="cosine",                 # 使用余弦学习率调度器
)

# 7. 初始化 SFTTrainer (Initialize SFTTrainer)
# ---
print("Initializing SFTTrainer...")
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=lora_config,
    dataset_text_field="text", # 数据集中包含文本的字段名
    max_seq_length=512,        # 最大序列长度
    tokenizer=tokenizer,
    args=training_arguments,
    packing=False,             # 是否将多个短样本打包成一个长样本
)

# 8. 开始训练 (Start Training)
# ---
print("Starting training...")
trainer.train()
print("Training finished.")

# 9. 保存模型 (Save Model)
# ---
print("Saving fine-tuned model...")
# 保存 LoRA 适配器
trainer.model.save_pretrained(new_model_path)

# 如果需要，合并模型并保存
# print("Merging and saving full model...")
# # 加载基础模型
# base_model = AutoModelForCausalLM.from_pretrained(
#     model_id,
#     low_cpu_mem_usage=True,
#     return_dict=True,
#     torch_dtype=torch.float16,
#     device_map="auto",
# )
# # 加载 LoRA 适配器并与基础模型合并
# merged_model = PeftModel.from_pretrained(base_model, new_model_path)
# merged_model = merged_model.merge_and_unload()

# # 保存合并后的完整模型和分词器
# merged_model.save_pretrained(new_model_path, safe_serialization=True)
# tokenizer.save_pretrained(new_model_path)
# print(f"Full model saved to {new_model_path}")

print("Fine-tuning process complete.")
