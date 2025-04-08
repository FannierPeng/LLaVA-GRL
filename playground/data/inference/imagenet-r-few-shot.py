from llava.eval.run_llava import eval_model_distrib,eval_model
import os
import json
from torch.utils.data import Dataset
import numpy as np
import torch
import random
# from transformers import CLIPImageProcessor
# from PIL import Image

# 配置路径
model_path = "/mnt/hdd/fpeng/LLaVA/checkpoints/llava-v1.5-7b"
imagenet_r_path = "/mnt/hdd/fpeng/LLaVA/playground/data/imagenet-r/"
support_path = "/mnt/hdd/fpeng/LLaVA/playground/data/imagenet/train/"
readme_path = "/mnt/hdd/fpeng/LLaVA/playground/data/imagenet-r/README.txt"
output_json_path = "/mnt/hdd/fpeng/LLaVA/playground/data/inference/imagenet_r_eval_results_3shot_1206v3.json"
prompt = "What is the picture of?"  ###"What is the category of the object in the picture?" ###
few_shot = 3

# 加载文件夹名到类名的映射
folder_to_classname = {}
with open(readme_path, "r") as f:
    i = 0
    for line in f:
        line = line.strip()
        if line and i <=200:
            folder_name, classname = line.split(maxsplit=1)
            folder_to_classname[folder_name] = classname.replace("_", " ")
            i+=1

# 初始化统计信息
results = []
correct_count = 0
total_count = 0
class_stats = {classname: {"correct": 0, "total": 0} for classname in folder_to_classname.values()}

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True

def collate_fn(support_class_to_images, few_shot, query_class):
    """
    自定义 collate 函数，每个 batch 包含 4 张图片：
    - 前三张图片来自 support  文件夹，属于不同类别；
    - 最后一张图片来自 query 文件夹，且类别与前三张不同。
    """
    # 从 support 中随机选择三个与query不同的类别
    support_classes = list(support_class_to_images.keys())
    support_classes = [cls for cls in support_classes if cls != query_class]
    random.shuffle(support_classes)
    selected_support_classes = support_classes[:few_shot]

    # 从每个类别中选择一张图片
    support_images = [random.choice(support_class_to_images[cls]) for cls in selected_support_classes]
    return support_images, selected_support_classes

class ImageFolderDataset(Dataset):
    def __init__(self, folder_to_classname):
        """
        Args:
            folder_to_classname (dict): Mapping from folder names to class names.
            query_template (str): Template for generating queries, e.g., "What is the picture of?"
            image_processor: Function or callable for preprocessing images.
        """
        self.support_data = []
        self.query_data = []
        self.support_class_to_images = {}
        self.query_class_to_images = {}
        # self.image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14-336")
        # i=0
        # 收集支持集的图片路径
        for folder_name, classname in folder_to_classname.items():
            folder_path = os.path.join(support_path, folder_name)

            if not os.path.isdir(folder_path):
                continue

            all_images = [
                os.path.join(folder_path, image_name)
                for image_name in os.listdir(folder_path)
                if image_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp'))
            ]
            # 随机取 3 张图片
            sampled_images = random.sample(all_images, min(few_shot, len(all_images)))
            self.support_data.extend([{"image_path": img, "classname": classname} for img in sampled_images])
            # 按类别存储
            self.support_class_to_images[classname] = sampled_images

        # 收集查询集的图片路径
        # i = 0
        for folder_name, classname in folder_to_classname.items():
            folder_path = os.path.join(imagenet_r_path, folder_name)
            # if i > 0:
            #     break
            if not os.path.isdir(folder_path):
                continue

            for image_name in os.listdir(folder_path):
                # if i >= 320:
                #     break
                image_file = os.path.join(folder_path, image_name)
                if not image_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp')):
                    continue
                else:
                    self.query_data.append({"image_path": image_file, "classname": classname})
                    if classname not in self.query_class_to_images:
                        self.query_class_to_images[classname] = []
                    self.query_class_to_images[classname].append(image_file)
                    # i+=1
            # i+=1

    def __len__(self):
        return len(self.query_data)

    def __getitem__(self, idx):
        query = self.query_data[idx]
        # image = Image.open(entry["image_path"]).convert("RGB")
        # image_tensor = self.image_processor(image)
        support_images, support_classes = collate_fn(self.support_class_to_images, few_shot, query["classname"])
        entry_images = support_images + [query["image_path"]]
        entry_classnames = support_classes + [query["classname"]]
        query = prompt
        # print('entry_images', entry_images)
        # print('entry_classnames', entry_classnames)
        return query, entry_images, entry_classnames


imagenet_r = ImageFolderDataset(folder_to_classname)

# 设置 eval_model 参数
args = type('Args', (), {
    "model_path": model_path,
    "model_base": None,
    "model_name": "llava-v1.5-7b",
    "query": prompt,
    "conv_mode": None,
    "sep": ",",
    "temperature": 0.2,
    "top_p": None,
    "num_beams": 1,
    "max_new_tokens": 512,
    "batch_size": 1,
    "num_gpus": 1,
    "few_shot": few_shot
})()
#
# 模型输出
results, class_acc = eval_model_distrib(args, imagenet_r)

for cls, it in class_acc.items():
    correct_count += it[0]
    total_count += it[1]

# 总准确率
accuracy = correct_count / total_count if total_count > 0 else 0
print(f"Overall Accuracy: {accuracy * 100:.2f}%")


# 保存结果为 JSON 文件
with open(output_json_path, "w") as f:
    json.dump({
        "results": results,
        "accuracy": accuracy,
        "class_acc": class_acc
    }, f, indent=4)

print(f"Evaluation results saved to {output_json_path}")
