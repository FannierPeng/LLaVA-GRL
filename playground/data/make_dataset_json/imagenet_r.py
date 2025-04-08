import os
import json

# ImageNet-R 文件夹路径
imagenet_r_path = "imagenet-r/"
real_imagenet_r_path = "/mnt/hdd/fpeng/LLaVA/playground/data/imagenet-r/"

# 输出 JSON 文件路径
output_json_path = "/mnt/hdd/fpeng/LLaVA/playground/data/imagenet-r.json"

# 初始化 dataset_content 列表
dataset_content = []

# 定义 README.txt 的路径
readme_path = "/mnt/hdd/fpeng/LLaVA/playground/data/imagenet-r/README.txt"

# 初始化文件夹名到类名的字典
folder_to_classname = {}

# 解析 README.txt 文件
with open(readme_path, "r") as f:
    i = 0
    for line in f:
        line = line.strip()  # 移除多余的空格和换行符
        if line and i <= 200:  # 跳过空行
            folder_name, classname = line.split(maxsplit=1)
            folder_to_classname[folder_name] = classname.replace("_", " ")
            i+=1


# 遍历 ImageNet-R 文件夹
for class_id in os.listdir(imagenet_r_path):
    class_path = os.path.join(imagenet_r_path, class_id)
    real_class_path = os.path.join(real_imagenet_r_path, class_id)
    if os.path.isdir(real_class_path):  # 只处理文件夹
        for image_name in os.listdir(real_class_path):
            image_path = os.path.join(class_path, image_name)
            entry = {
                "id": f"imagenet_r-{os.urandom(4).hex()}",
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

print('dataset_content: ',len(dataset_content))
# 将数据写入 JSON 文件
with open(output_json_path, "w") as f:
    json.dump(dataset_content, f)

print(f"Dataset JSON written to {output_json_path}")


