import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from peft import PeftModel

# 1. 配置 (Configuration)
# ---
# 基础模型ID
base_model_id = "google/gemma-3-270m"
# LoRA 适配器路径 (我们微调后保存的路径)
lora_adapter_path = "./gemma-3-270m-teaching-eval"

# 2. 加载分词器和模型 (Load Tokenizer and Model)
# ---
print("Loading base model and tokenizer...")
# 加载分词器
tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)

# 加载基础模型
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    low_cpu_mem_usage=True,
    return_dict=True,
    torch_dtype=torch.float16,
    device_map="auto",
)

# 3. 应用 LoRA 适配器 (Apply LoRA Adapter)
# ---
print(f"Loading LoRA adapter from {lora_adapter_path}...")
# 加载 PeftModel，它会自动将 LoRA 模块合并到基础模型中
model = PeftModel.from_pretrained(base_model, lora_adapter_path)
# 将模型设置为评估模式
model = model.eval()

print("Model loaded successfully.")

# 4. 创建推理 Pipeline (Create Inference Pipeline)
# ---
pipe = pipeline(
    task="text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=512, # 控制生成文本的最大长度
    repetition_penalty=1.1 # 添加重复惩罚
)

# 5. 准备输入并进行推理 (Prepare Input and Run Inference)
# ---
# 准备一段新的教学转录文本进行评估
new_transcript = (
    "“今天我们学习光合作用。光合作用呢，就是植物在光下，用二氧化碳和水，"
    "合成有机物，然后放出氧气的过程。公式是 CO2 + H2O -> C6H12O6 + O2。"
    "条件是光照和叶绿体。这个很重要，考试要考的，大家背下来。好，就是这些内容。”"
)

# 构建符合训练格式的 Prompt
prompt = (
    f"<s>[INST] 指令：请根据以下师范生的课堂教学转录文本，从知识点、讲解逻辑性、"
    f"术语专业性、教学指令四个方面进行详细评估，并给出改进建议。\n\n"
    f"转录文本：\n“{new_transcript}” [/INST] \n\n"
)

print("\n--- Running Inference ---\n")
print(f"Input Transcript:\n{new_transcript}\n")

# 使用 pipeline 进行推理
result = pipe(prompt)

# 6. 输出结果 (Print Result)
# ---
# 结果是一个列表，我们取出第一个元素的生成文本
generated_text = result[0]['generated_text']
# 为了美观，我们只打印模型生成的部分（即 [/INST] 之后的部分）
response = generated_text.split("[/INST]")[1].strip()

print(f"--- Model's Evaluation ---\n")
print(response)
print("\n--- Inference Complete ---")
