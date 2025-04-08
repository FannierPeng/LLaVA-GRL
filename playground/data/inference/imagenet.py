from llava.eval.run_llava import eval_model_distrib,eval_model
import os
import json
from torch.utils.data import Dataset
# from transformers import CLIPImageProcessor
# from PIL import Image

# 配置路径
model_path = "/mnt/hdd/fpeng/LLaVA/checkpoints/llava-v1.5-7b-merged_imagenet_4shot_mask0d25-RL-routers_1"
imagenet_path = "/mnt/hdd/fpeng/LLaVA/playground/data/imagenet/val/"
readme_path = "/mnt/hdd/fpeng/LLaVA/playground/data/imagenet_classnames.txt"
output_json_path = "/mnt/hdd/fpeng/LLaVA/playground/data/inference/imagenet_eval_results_ftlora-4shot-mask0d25-RL-routers_1.json"
prompt = "What is the picture of?" ###"What is the picture of?"

# 加载文件夹名到类名的映射
folder_to_classname = {}
with open(readme_path, "r") as f:
    for line in f:
        line = line.strip()
        if line:
            folder_name, classname = line.split(maxsplit=1)
            folder_to_classname[folder_name] = classname.replace("_", " ")
    f.close()

# 初始化统计信息
results = []
correct_count = 0
total_count = 0
class_stats = {classname: {"correct": 0, "total": 0} for classname in folder_to_classname.values()}
class_names = folder_to_classname.values()

class ImageFolderDataset(Dataset):
    def __init__(self, folder_to_classname):
        """
        Args:
            folder_to_classname (dict): Mapping from folder names to class names.
            query_template (str): Template for generating queries, e.g., "What is the picture of?"
            image_processor: Function or callable for preprocessing images.
        """
        self.data = []
        # self.image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14-336")
        # i=0
        for folder_name, classname in folder_to_classname.items():
            folder_path = os.path.join(imagenet_path, folder_name)
            # if i > 10:
            #     break
            if not os.path.isdir(folder_path):
                continue

            for image_name in os.listdir(folder_path):
                image_file = os.path.join(folder_path, image_name)
                # print(image_file)
                if not image_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp')):
                    continue  # 跳过非图片文件
                else:
                    self.data.append({
                        "image_path": image_file,
                        "classname": classname
                    })
                    # i+=1

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        entry = self.data[idx]
        # image = Image.open(entry["image_path"]).convert("RGB")
        # image_tensor = self.image_processor(image)
        query = prompt
        return query, entry["image_path"], entry["classname"]

imagenet = ImageFolderDataset(folder_to_classname)

# 设置 eval_model 参数
args = type('Args', (), {
    "model_path": model_path,
    "model_base": None,
    "model_name": "llava-v1.5-7b",
    "query": prompt,
    "conv_mode": None,
    "sep": ",",
    "temperature": 0,
    "top_p": None,
    "num_beams": 1,
    "max_new_tokens": 512,
    "batch_size": 8,
    "num_gpus": 1,
    "few_shot": 0,
    "log_probs": False
})()
#
# 模型输出
results, class_acc = eval_model_distrib(args, imagenet, class_names)


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
