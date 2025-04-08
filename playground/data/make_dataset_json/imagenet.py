import os
import json
import torch
import numpy as np
import random
import pickle

# ImageNet 文件夹路径
imagenet_path = "imagenet/"
real_imagenet_path = "/mnt/hdd/fpeng/LLaVA/playground/data/imagenet/train/"
fewshot_datapath = '/mnt/hdd/fpeng/LLaVA/playground/data/split_fewshot/imagenet_split_fewshot/shot_4-seed_42.pkl'

# 输出 JSON 文件路径
output_json_path = "/mnt/hdd/fpeng/LLaVA/playground/data/imagenet-train-4shot.json"

# 初始化 dataset_content 列表
dataset_content = []

# 定义 README.txt 的路径
readme_path = "/mnt/hdd/fpeng/LLaVA/playground/data/imagenet_classnames.txt"

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True

# 初始化文件夹名到类名的字典
folder_to_classname = {}

with open(fewshot_datapath, 'rb') as file:
    fewshot_data = pickle.load(file)
with open(fewshot_datapath, "rb") as file:
    fewshot_data = pickle.load(file)
    fewshot_data_train = fewshot_data["train"]
# labels = [item.label for item in fewshot_data_train]
impaths = [item.impath.replace('/userhome/CLIP/data/imagenet/images/train/',real_imagenet_path) for item in fewshot_data_train]
# classnames = [item.classname for item in fewshot_data_train]
#
with open(readme_path, "r") as f:
    for line in f:
        line = line.strip()  # 移除多余的空格和换行符
        if line:  # 跳过空行
            folder_name, classname = line.split(maxsplit=1)
            folder_to_classname[folder_name] = classname.replace("_", " ")


# 遍历 ImageNet-train 文件夹
for class_id in os.listdir(real_imagenet_path):
    class_path = os.path.join(imagenet_path+'/train/', class_id)
    real_class_path = os.path.join(real_imagenet_path, class_id)
    if os.path.isdir(real_class_path):  # 只处理文件夹
        for image_name in os.listdir(real_class_path):
            image_path = os.path.join(class_path, image_name)
            real_image_path = os.path.join(real_class_path, image_name)
            if real_image_path in impaths:
                entry = {
                    "id": f"imagenet_-{os.urandom(4).hex()}",
                    "image": image_path,
                    "conversations": [
                        {
                            "from": "human",
                            "value": "<image>\nWhat is the picture of?"
                        },
                        {
                            "from": "gpt",
                            "value": f"{folder_to_classname[class_id]}."
                        }
                    ]
                }

                dataset_content.append(entry)

print('dataset_content: ',len(dataset_content)) #1000
# 将数据写入 JSON 文件
with open(output_json_path, "w") as f:
    json.dump(dataset_content, f)

print(f"Dataset JSON written to {output_json_path}")

