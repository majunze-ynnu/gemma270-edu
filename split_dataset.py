import json
import random

def split_dataset(
    input_file="teaching_eval_dataset.jsonl",
    train_output_file="train_dataset.jsonl",
    eval_output_file="eval_dataset.jsonl",
    split_ratio=0.9
):
    """
    读取一个JSONL格式的数据集文件，将其随机打乱并划分为训练集和验证集。

    Args:
        input_file (str): 输入的JSONL文件名。
        train_output_file (str): 输出的训练集文件名。
        eval_output_file (str): 输出的验证集文件名。
        split_ratio (float): 训练集所占的比例。
    """
    # 1. 读取所有数据
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            all_data = [json.loads(line) for line in f]
    except FileNotFoundError:
        print(f"错误: 输入文件 '{input_file}' 未找到。")
        print("请先运行 generate_dataset.py 来生成数据集。")
        return

    # 2. 随机打乱数据
    random.shuffle(all_data)

    # 3. 计算切分点
    num_total = len(all_data)
    num_train = int(num_total * split_ratio)

    # 4. 切分数据集
    train_data = all_data[:num_train]
    eval_data = all_data[num_train:]

    # 5. 写入训练集文件
    with open(train_output_file, 'w', encoding='utf-8') as f:
        for entry in train_data:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

    # 6. 写入验证集文件
    with open(eval_output_file, 'w', encoding='utf-8') as f:
        for entry in eval_data:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

    print(f"数据集划分完成。")
    print(f"总数据量: {num_total}")
    print(f"训练集数量: {len(train_data)} (保存至 {train_output_file})")
    print(f"验证集数量: {len(eval_data)} (保存至 {eval_output_file})")


if __name__ == "__main__":
    split_dataset()
