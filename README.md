# Gemma-3-270M 教学能力评估模型：微调与评估流水线

本项目提供了一整套完整的代码流水线（Pipeline），用于微调、评估并可视化分析一个基于 Google Gemma-3-270M 的教学能力评估模型。

## 项目特色

*   **端到端流程**: 包含了从数据生成、数据集划分、模型微调、推理验证到结果可视化的完整流程。
*   **综合评估**: 采用多种业界标准指标（Perplexity, ROUGE, BLEU）对模型性能进行全方位、可量化的评估。
*   **可视化分析**: 集成 TensorBoard，可以方便地通过图表来监控和分析训练过程中的各项指标变化。
*   **高度可定制**: 提供了数据生成脚本，方便用户根据自己的需求扩展数据集。

## 文件结构

```
.
├── generate_dataset.py           # 1. 用于程序化生成教学评估数据集的脚本
├── split_dataset.py              # 2. 用于将数据集划分为训练集和验证集的脚本
├── finetune_gemma.py             # 3. 执行模型微调和评估的核心脚本
├── inference.py                  # 4. 使用微调后的模型进行推理的脚本
├── teaching_eval_dataset.jsonl   # (生成) 完整的原始数据集
├── train_dataset.jsonl           # (生成) 训练专用数据集
├── eval_dataset.jsonl            # (生成) 验证专用数据集
└── README.md                     # 本指南文件
```

---

## 完整操作流程

### 步骤一：环境设置和依赖安装

1.  **创建虚拟环境 (推荐)**:
    ```bash
    python -m venv gemma-finetune-env
    source gemma-finetune-env/bin/activate  # On Windows, use `gemma-finetune-env\Scripts\activate`
    ```

2.  **安装所有必要的库**:
    ```bash
    pip install torch transformers datasets peft trl bitsandbytes accelerate evaluate rouge_score tensorboard nltk
    ```

3.  **登录 Hugging Face Hub**:
    Gemma 是受限模型，请确保您已获得访问权限。
    ```bash
    huggingface-cli login
    ```

### 步骤二：准备数据集

1.  **生成原始数据集**:
    运行脚本以生成一个包含200+条模拟数据的大数据集。
    ```bash
    python generate_dataset.py
    ```

2.  **划分数据集**:
    运行脚本将上一步生成的数据集划分为训练集和验证集。
    ```bash
    python split_dataset.py
    ```
    运行后，你将得到 `train_dataset.jsonl` 和 `eval_dataset.jsonl` 两个文件。

### 步骤三：执行模型微调与评估

运行核心微调脚本。此脚本会自动加载训练集和验证集，进行模型训练，并在每个训练周期（Epoch）结束时进行一次评估。
```bash
python finetune_gemma.py
```
训练过程中，您会在终端日志中看到类似以下的评估结果：
```
{'eval_loss': 1.85, 'eval_rouge2': 0.23, 'eval_rougeL': 0.45, 'eval_bleu': 0.31, 'eval_runtime': 60.0, ...}
```

### 步骤四：可视化分析训练过程

训练脚本会自动将所有指标（包括损失、ROUGE、BLEU等）记录下来。训练结束后，运行以下命令启动 TensorBoard：
```bash
tensorboard --logdir=./results/runs
```
在浏览器中打开显示的网址（通常是 `http://localhost:6006/`），您就可以看到所有指标随训练变化的曲线图。

### 步骤五：进行推理测试

当您对训练结果满意后，可以使用 `inference.py` 脚本来对新的教学文本进行定性评估，直观地感受模型的效果。
```bash
python inference.py
```

---

## 评估指标解读

*   **Loss (损失)**: 训练和验证过程中的损失值。**越低越好**，是模型优化的核心目标。
*   **Perplexity (困惑度)**: `eval_loss` 的指数形式 (`exp(eval_loss)`)。它衡量模型对文本的预测能力。**越低越好**。
*   **ROUGE**: 衡量模型生成文本与参考答案在内容上的重合度。**分数越高越好**。
*   **BLEU**: 衡量模型生成文本与参考答案在流畅度和词组上的相似度。**分数越高越好**。
*   **`load_best_model_at_end`**: 脚本配置了此项，意味着最终保存下来供推理使用的，将是训练过程中在验证集上 `eval_loss` 最低（即表现最好）的那个模型版本。
