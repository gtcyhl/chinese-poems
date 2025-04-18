from torch.optim import Adam
from transformers.optimization import get_scheduler
import torch
from data import ChineseDataset
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained(
    r"E:\github_rep\chinese-poems\model\gpt2-chinese-cluecorpussmall"
)
model = AutoModelForCausalLM.from_pretrained(
    r"E:\github_rep\chinese-poems\model\gpt2-chinese-cluecorpussmall"
)
# print(model)

dataset = ChineseDataset()
def collate_fn(data):

    data = tokenizer.batch_encode_plus(
        data,
        padding=True,  # 填充序列
        truncation=True,  # 截断序列
        max_length=512,  # 最大序列长度
        return_tensors="pt",
    ) 
    data["labels"] = data["input_ids"].clone()   # 创建标签，与输入ID相同（生成模型，输入也作为标签）
    return data

loader = torch.utils.data.DataLoader( # 创建数据加载器，用于批量加载数据
    dataset=dataset,  # 指定数据集
    batch_size=6,  # 指定批量大小
    collate_fn=collate_fn,  # 指定预处理函数
    shuffle=True,  # 打乱数据
    drop_last=True,  # 如果最后一个批次不足，则丢弃
)
print(len(loader)) # 打印数据加载器中的批次数量

def train():
    global model  # 使用全局变量model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    optimizer = Adam(model.parameters(), lr=2e-5)
    scheduler = get_scheduler(
        name="linear",  # 线性调度器
        num_warmup_steps=0,  # 预热步数
        num_training_steps=len(loader),  # 总训练步数
        optimizer=optimizer,
    )

    model.train()
    for i, data in enumerate(loader):
        for k in data.keys():
            data[k] = data[k].to(device)

        out = model(**data)
        loss = out["loss"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # 梯度裁剪，防止梯度爆炸

        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        model.zero_grad()

        if i % 50 == 0:
            # 准备标签和输出用于计算准确率
            labels = data["labels"][:, 1:]
            # 通过‘logits’获取模型的原始输出值
            out = out["logits"].argmax(dim=2)[:, :-1]

            # 移除在数据预处理阶段添加的填充（通常是0），以便只计算实际数据部分的损失和准确率，避免填充部分对模型性能评估的影响。
            select = labels != 0
            labels = labels[select]
            out = out[select]
            del select

            # 计算准确率
            accuracy = (labels == out).sum().item() / labels.numel()

            # 获取当前学习率
            lr = optimizer.state_dict()["param_groups"][0]["lr"]

            # 打印批次索引、损失、学习率和准确率
            print(i, loss.item(), lr, accuracy)
   
    torch.save(model.state_dict(), "net.pt")
    print("权重保存成功！")

if __name__ == "__main__":
    for epoch in range(1):
        train()
