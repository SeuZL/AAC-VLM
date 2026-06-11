
import os
import json
import torch
import torch.nn as nn
import torch_npu
import timm
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score
import warnings
warnings.filterwarnings("ignore")

# ==== 配置参数 ====
jsonl_file = "/root/zhangliang/Qwen2.5-VL/yanke/multi_label_data.jsonl"
batch_size = 16
num_epochs = 50
lr = 1e-4
image_size = 224
device = torch.device("npu:0")
print(f"✅ Using NPU device: {device}")

# ==== 标签列表（固定 40 类） ====
label_list = [
    "窥不入", "影动", "黄斑区", "中心凹", "视盘", "动脉", "静脉", "毛细血管", "网膜下遮蔽", "后极部", 
    "中周部", "眼周边", "拱环", "高荧光", "低荧光", "荧光着染", "遮蔽荧光", "积存", "色素上皮", 
    "血管壁染", "荧光渗漏", "透见荧光", "点状", "片状", "簇样", "点片状", "微血管瘤样", "不规则", 
    "弥漫样", "息肉样", "变性区", "异常血管", "萎缩灶", "激光斑", "出血", "呈高度近视眼底改变", 
    "网膜漂浮", "NP", "NV", "NVD"
]
label2idx = {label: i for i, label in enumerate(label_list)}
num_classes = len(label_list)
print(f"✅ 标签数量: {num_classes} 类")

# ==== 数据集定义 ====
class MultiLabelDataset(Dataset):
    def __init__(self, jsonl_path, transform=None):
        self.data = []
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                self.data.append(json.loads(line))
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        img_path = item['image']
        labels = item['labels']

        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        target = torch.zeros(num_classes)
        for lbl in labels:
            if lbl in label2idx:
                target[label2idx[lbl]] = 1.0

        return image, target

# ==== 图像预处理 ====
transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor()
])

# ==== 数据加载器 ====
dataset = MultiLabelDataset(jsonl_file, transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# ==== 构建模型 ====
model = timm.create_model('efficientnet_b0', pretrained=False)
state_dict = torch.load("/root/zhangliang/Qwen2.5-VL/yanke/pytorch_model.bin", map_location="cpu")
model.load_state_dict(state_dict)

model.classifier = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(model.classifier.in_features, num_classes),
    nn.Sigmoid()
)
model = model.to(device)

# ==== 损失函数与优化器 ====

criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr, foreach=False)


# ==== 模型训练 ====
best_f1 = 0
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    y_true, y_pred = [], []

    for imgs, targets in dataloader:
        imgs, targets = imgs.to(device), targets.to(device)
        outputs = model(imgs)
        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        y_true.extend(targets.cpu().numpy())
        y_pred.extend((outputs > 0.5).float().cpu().numpy())

    f1 = f1_score(y_true, y_pred, average='micro')
    print(f"📘 Epoch {epoch+1}/{num_epochs} | Loss: {total_loss:.4f} | Micro F1: {f1:.4f}")

    if f1 > best_f1:
        best_f1 = f1
        torch.save(model.state_dict(), "best_model.pth")
        print("💾 最优模型已保存: best_model.pth")

print("🎉 训练完成！Best F1-score:", best_f1)
