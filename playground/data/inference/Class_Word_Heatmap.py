##Class Word Heatmap
import random
import numpy as np
from collections import Counter, defaultdict
import json
import matplotlib.pyplot as plt
import seaborn as sns


result_json_path = "imagenet_eval_results_ftlora-lr2efu5.json"
readme_path = "../imagenet_classnames.txt"

random.seed(42)
num_classes = 1000  # 类的数量
samples_per_class = 50  # 每个类50个样本

# 类标签
folder_to_classname = {}
with open(readme_path, "r") as f:
    for line in f:
        line = line.strip()
        if line:
            folder_name, classname = line.split(maxsplit=1)
            folder_to_classname[folder_name] = classname.replace("_", " ")

# 读取 JSON 文件
with open(result_json_path, "r") as f:
    data = json.load(f)

# 访问数据内容
results = data["results"]       # 模型预测的结果 list
accuracy = data["accuracy"]     # 总准确率  .4f
class_acc = data["class_acc"]   # 每个类的准确率  dict

# 统计每个类的预测词分布
class_predictions = defaultdict(list)

# 遍历 results 数据
for result in results:
    true_label = result["true_label"]  # 标签类名
    prediction = result["predicted_label"].strip().strip('.').lower()  # 预测词
    class_predictions[true_label].append(prediction)

# 统计预测词频率
top_k = 10
class_prediction_counts = {label: Counter(predictions).most_common(top_k) for label, predictions in class_predictions.items()}

'''
Class: cat
  Prediction: cat -> Count: 2
  Prediction: dog -> Count: 1
  Prediction: bird -> Count: 1
Class: bird
  Prediction: bird -> Count: 1
Class: dog
  Prediction: dog -> Count: 2
'''

# 3. 绘制词热力图
def plot_word_heatmap(class_prediction_counts):
    """绘制一个类的预测词热力图，显示前 top_k 个出现最多的预测词"""
    for label, prediction_count in class_prediction_counts.items():
        # 数据准备：将预测词和出现次数转换成散点图的坐标和大小
        predictions = [word for word, count in prediction_count]
        counts = [count for word, count in prediction_count]

        plt.figure(figsize=(10, 6))  # 设置图形大小
        plt.title(f"Top-{top_k} Prediction Heatmap for Class: {label}", fontsize=16)
        plt.xlabel("Prediction Words", fontsize=12)
        plt.ylabel("Count (Frequency)", fontsize=12)

        # 绘制散点图，圆圈大小由预测词频率决定
        for i, (word, count) in enumerate(prediction_count):
            plt.scatter(i, count, s=count * 500, alpha=0.5, edgecolors="k", label=word)  # s 控制圆圈大小

            # 在圆圈中显示预测词
            plt.text(i, count, word, fontsize=10, ha='center', va='center', color='white')

        # 设置 x 轴标签为预测词
        plt.xticks(range(len(predictions)), predictions, rotation=45, fontsize=10)
        plt.yticks(fontsize=10)

        plt.tight_layout()
        # plt.show()
        plt.savefig(f"./imagenet_eval_results/heatmap_{label}.png", dpi=300)

# 调用函数，绘制每个类的预测词热力图
plot_word_heatmap(class_prediction_counts)


