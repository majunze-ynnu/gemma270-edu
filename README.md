# 使用 Gemma-3-270M 微调教学能力评估模型

本项目提供了一整套代码，用于微调 Google 的 Gemma-3-270M 模型，使其能够根据师范生的课堂教学转录文本，评估其教学能力并提供改进建议。

## 项目目标

通过对一个专门构建的“师范生教学能力评估”数据集进行微调，使 Gemma-3-270M 模型学习到如何从四个维度（知识点、讲解逻辑性、术语专业性、教学指令）分析教学文本，并生成结构化的评估报告。

## 文件结构

```
.
├── teaching_eval_dataset.jsonl   # 用于微调的合成数据集
├── finetune_gemma.py             # 执行模型微调的 Python 脚本
├── inference.py                  # 使用微调后的模型进行推理的 Python 脚本
└── README.md                     # 本指南文件
```

---

## 步骤一：环境设置和依赖安装

1.  **创建虚拟环境 (推荐)**:
    ```bash
    python -m venv gemma-finetune-env
    source gemma-finetune-env/bin/activate  # On Windows, use `gemma-finetune-env\Scripts\activate`
    ```

2.  **安装必要的库**:
    你需要安装 PyTorch 以及 Hugging Face 生态系统中的几个关键库。
    ```bash
    pip install torch transformers datasets peft trl bitsandbytes accelerate
    ```
    *   `torch`: 深度学习框架。
    *   `transformers`: Hugging Face 的核心库，用于加载模型和分词器。
    *   `datasets`: 用于加载我们创建的 `jsonl` 数据集。
    *   `peft`: 参数高效微调库 (Parameter-Efficient Fine-Tuning)，用于实现 LoRA。
    *   `trl`: Transformer 强化学习库，提供了方便的 `SFTTrainer`。
    *   `bitsandbytes`: 用于实现模型量化，节省显存。
    *   `accelerate`: 简化 PyTorch 在不同硬件上训练的工具。

3.  **登录 Hugging Face Hub**:
    Gemma 模型是受限模型，你需要先在 Hugging Face 网站上申请访问权限。之后，在你的终端运行以下命令并输入你的 Access Token 进行登录。
    ```bash
    huggingface-cli login
    ```

---

## 步骤二：执行模型微调

在安装完所有依赖并登录成功后，运行微调脚本。

```bash
python finetune_gemma.py
```

这个过程会从 Hugging Face Hub 下载 Gemma-3-270M 模型（FP16 格式约 540MB），然后加载 `teaching_eval_dataset.jsonl` 文件进行微调。根据你的硬件配置（尤其是 GPU），这可能需要一些时间。

训练完成后，一个名为 `gemma-3-270m-teaching-eval` 的文件夹会被创建，其中包含了微调后的 LoRA 适配器权重。

---

## 步骤三：进行推理

微调完成后，你可以使用 `inference.py` 脚本来测试模型的效果。

```bash
python inference.py
```

该脚本会：
1.  加载原始的 Gemma-3-270M 模型。
2.  加载并应用保存在 `gemma-3-270m-teaching-eval` 里的 LoRA 适配器。
3.  对脚本中预设的一段新的教学文本进行评估。
4.  在终端打印出完整的评估报告。

---

## 如何使用你自己的数据

1.  **打开 `teaching_eval_dataset.jsonl` 文件**。
2.  **模仿现有格式，添加你自己的数据**。每一行都是一个独立的 JSON 对象，包含一个 `text` 字段。`text` 字段的内容必须遵循 `<s>[INST] ... [/INST] ... </s>` 的格式。
3.  **重新运行 `finetune_gemma.py`** 来基于你的新数据进行微调。数据越多，模型的效果通常越好。
