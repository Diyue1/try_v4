import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import average_precision_score, accuracy_score
from model_rswa_v4 import AIGCDetector

# 路径与超参数对齐论文 [cite: 221, 245, 246]
TRAIN_DIR = "./train" # 包含 4-class 设置: car, cat, chair, horse
VAL_DIR = "./val"
BATCH_SIZE = 16
ACCUMULATION_STEPS = 8
LR = 2e-4
EPOCHS = 90
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BinaryDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.samples = []
        self.transform = transform
        # 论文特定的 4-class 训练设置 [cite: 221]
        target_cats = ['car', 'cat', 'chair', 'horse']
        for root, _, files in os.walk(root_dir):
            if any(cat in root for cat in target_cats):
                if '0_real' in root or '1_fake' in root:
                    label = 0 if '0_real' in root else 1
                    for f in files:
                        if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                            self.samples.append((os.path.join(root, f), label))
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert('RGB')
        if self.transform: img = self.transform(img)
        return img, label
def train():
    # 数据增强保持不变
    train_tf = transforms.Compose([
        transforms.RandomCrop((256, 256), pad_if_needed=True),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_tf = transforms.Compose([
        transforms.CenterCrop((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_loader = DataLoader(BinaryDataset(TRAIN_DIR, train_tf), batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(BinaryDataset(VAL_DIR, val_tf), batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    model = AIGCDetector().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.BCEWithLogitsLoss()
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)
    
    # 【新增】：混合精度训练器
    scaler = torch.amp.GradScaler('cuda') [cite: 14]

    best_acc = 0.0
    for epoch in range(EPOCHS):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        optimizer.zero_grad()
        
        for i, (inputs, labels) in enumerate(pbar):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE).float()
            
            # 【修改】：开启自动混合精度
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss = loss / ACCUMULATION_STEPS # 缩放损失
            
            scaler.scale(loss).backward()

            # 每积累 8 个 batch 更新一次参数
            if (i + 1) % ACCUMULATION_STEPS == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
            pbar.set_postfix(loss=loss.item() * ACCUMULATION_STEPS)

        scheduler.step()

        # --- 修正后的验证部分 --- 
        model.eval()
        y_true, y_prob = [], []
        with torch.no_grad():
            for inputs, labels in val_loader:
                # 使用最新的 torch.amp 语法消除警告 
                with torch.amp.autocast('cuda'):
                    logits = model(inputs.to(DEVICE))
                    # 手动补上 sigmoid [cite: 13]
                    probs = torch.sigmoid(logits)
                
                y_true.extend(labels.numpy())
                # 【关键修复】：将 outputs 修改为 probs 
                y_prob.extend(probs.cpu().numpy())

        # 计算指标
        y_prob = np.array(y_prob)
        y_true = np.array(y_true)
        y_pred = (y_prob > 0.5).astype(int)
        
        acc = accuracy_score(y_true, y_pred) * 100
        ap = average_precision_score(y_true, y_prob) * 100
        print(f"Epoch {epoch+1} | Acc: {acc:.2f}% | AP: {ap:.2f}%")

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), "best_model_paper.pth")

if __name__ == "__main__":
    train()


