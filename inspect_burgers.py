import torch
from pathlib import Path

# 定义文件路径
file_path = (
    "/Users/cylenlc/work/neuraloperator/neuralop/data/datasets/data/burgers_train_16.pt"
)

# 加载数据
data = torch.load(file_path)

# 查看数据结构
print(f"数据类型: {type(data)}")

if isinstance(data, dict):
    print("包含的键:", data.keys())
    for key, value in data.items():
        if isinstance(value, torch.Tensor):
            print(
                f"键 '{key}': 类型={type(value)}, 形状={value.shape}, 数据类型={value.dtype}"
            )
        else:
            print(f"键 '{key}': 类型={type(value)}")
else:
    print(f"内容概要: {data}")
