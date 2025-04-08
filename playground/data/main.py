import json
import mindspore as ms
from mindspore import nn, Tensor, Model
from mindspore.dataset import GeneratorDataset
from mindspore.train.callback import LossMonitor
from transformers import BertTokenizer

# 数据集加载函数
class FactDataset:
    def __init__(self, data_path, is_training=True):
        self.data = self.load_data(data_path)
        self.is_training = is_training

    def load_data(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def __getitem__(self, idx):
        item = self.data[idx]
        question = item["Question"]
        answer = item["Answer"]
        label = 1 if self.is_training and item["Hallucination"] == "YES" else 0
        return question, answer, label

    def __len__(self):
        return len(self.data)

# 数据预处理
def preprocess(data, tokenizer, max_length=128):
    question, answer, label = data
    input_text = f"Question: {question} Answer: {answer}"
    encoding = tokenizer(input_text, truncation=True, max_length=max_length, padding='max_length', return_tensors="ms")
    return encoding["input_ids"], encoding["attention_mask"], Tensor(label, ms.int32)

# 数据集生成器
def create_dataset(data_path, tokenizer, batch_size, is_training=True):
    dataset = FactDataset(data_path, is_training=is_training)
    preprocess_fn = lambda data: preprocess(data, tokenizer)
    generator = GeneratorDataset(dataset, column_names=["input_ids", "attention_mask", "label"], shuffle=is_training)
    return generator.map(preprocess_fn).batch(batch_size)

class FactConsistencyModel(nn.Cell):
    def __init__(self, pretrained_model):
        super(FactConsistencyModel, self).__init__()
        self.bert = pretrained_model
        self.classifier = nn.Dense(768, 2)  # 二分类

    def construct(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output  # [CLS] token 表示
        logits = self.classifier(pooled_output)
        return logits

def train_model(train_data, val_data, model, tokenizer, epochs=3, lr=1e-4, batch_size=16):
    loss_fn = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")
    optimizer = nn.Adam(model.trainable_params(), learning_rate=lr)
    model = Model(model, loss_fn=loss_fn, optimizer=optimizer, metrics={"accuracy"})

    train_ds = create_dataset(train_data, tokenizer, batch_size, is_training=True)
    val_ds = create_dataset(val_data, tokenizer, batch_size, is_training=False)

    model.train(epochs, train_ds, callbacks=[LossMonitor()], dataset_sink_mode=False)
    metrics = model.eval(val_ds, dataset_sink_mode=False)
    print("Validation Accuracy:", metrics["accuracy"])
    return model


def predict(test_data, model, tokenizer, batch_size=16):
    test_ds = create_dataset(test_data, tokenizer, batch_size, is_training=False)
    predictions = []

    for data in test_ds.create_dict_iterator():
        logits = model.predict(data["input_ids"], data["attention_mask"])
        predicted = logits.argmax(axis=1).asnumpy()
        for i, pred in enumerate(predicted):
            result = {
                "Question": data["Question"][i],
                "Answer": data["Answer"][i],
                "Prediction": "YES" if pred == 1 else "NO"
            }
            predictions.append(result)

    return predictions

# 保存预测结果
def save_predictions(predictions, output_path):
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(predictions, f, ensure_ascii=False, indent=4)


from transformers import BertModel

if __name__ == "__main__":
    # 参数设置
    train_data_path = "factuality_train.json"
    test_data_path = "factuality_test.json"
    output_path = "factuality_predict.json"
    model_path = "pengcheng-mind-7b"
    tokenizer = BertTokenizer.from_pretrained(model_path)

    # 加载模型
    pretrained_model = BertModel.from_pretrained(model_path)
    model = FactConsistencyModel(pretrained_model)

    # 模型训练
    trained_model = train_model(train_data_path, train_data_path, model, tokenizer)

    # 推理预测
    predictions = predict(test_data_path, trained_model, tokenizer)

    # 保存结果
    save_predictions(predictions, output_path)
