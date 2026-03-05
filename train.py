import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image, ImageFile
from tqdm import tqdm
from sklearn.metrics import average_precision_score, accuracy_score
import numpy as np
import random
import io  # 新增：用于JPEG压缩

from model_rswa import AIGCDetector

ImageFile.LOAD_TRUNCATED_IMAGES = True

# 路径配置
TRAIN_DIR = "/data/ziqiang/yjz/dataset/Benchmark/newTrain/train"
VAL_DIR = "/data/ziqiang/yjz/dataset/Benchmark/newTrain/val"

PHYSICAL_BATCH_SIZE = 64
TARGET_BATCH_SIZE = 128
ACCUM_STEPS = TARGET_BATCH_SIZE // PHYSICAL_BATCH_SIZE

LR = 2e-4
EPOCHS = 20
NUM_WORKERS = 2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class RandomJPEGCompression:
    def __init__(self, quality_min=60, quality_max=100, p=0.5):
        self.quality_min = quality_min
        self.quality_max = quality_max
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            quality = random.randint(self.quality_min, self.quality_max)
            img = img.convert('RGB')
            with io.BytesIO() as buffer:  # 使用 with 上下文管理
                img.save(buffer, "JPEG", quality=quality)
                buffer.seek(0)
                new_img = Image.open(buffer)
                new_img.load()  # 强制加载像素数据
                return new_img
        return img


class RecursiveBinaryDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        self.target_categories = ['car', 'cat', 'chair', 'horse']

        if not os.path.exists(root_dir):
            raise RuntimeError(f"路径不存在: {root_dir}")

        for root, dirs, files in os.walk(root_dir):
            parts = root.split(os.sep)
            category_found = any(cat in parts for cat in self.target_categories)

            if not category_found:
                continue

            folder_name = os.path.basename(root)
            if folder_name in ['0_real', '1_fake']:
                label = 0 if folder_name == '0_real' else 1
                for file in files:
                    if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        self.samples.append((os.path.join(root, file), label))

        # 打乱数据，防止按文件夹顺序读取导致训练不稳
        random.shuffle(self.samples)

        if len(self.samples) == 0:
            raise RuntimeError(f"未找到数据！请检查路径结构。")
        print(f"[Dataset] {root_dir}: {len(self.samples)} images loaded.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            image = Image.open(path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, label
        except Exception as e:
            print(f"Warning: Failed to load image {path}: {e}")
            # 返回全黑图作为 fallback，防止训练中断
            fallback = torch.zeros((3, 256, 256))
            return fallback, label


def train_model():
    print(f"环境: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

    train_transform = transforms.Compose([
        # 1. 随机裁剪：严禁使用 Resize，以保留高频伪影
        transforms.RandomCrop((256, 256), pad_if_needed=True, padding_mode='reflect'),

        # 2. 几何增强：翻转与旋转
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),

        # 3. [补全缺口] 通用性增强：模拟网络传播和不同生成器的特征
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))], p=0.2),  # 模拟模糊
        RandomJPEGCompression(quality_min=50, quality_max=100, p=0.5),  # 模拟压缩
        transforms.ColorJitter(brightness=0.1, contrast=0.1),  # 颜色微扰

        # 4. 转张量与归一化
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_transform = transforms.Compose([
        # 验证时取图片中心 256x256，保证输入一致性且不破坏频率特征
        transforms.CenterCrop((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    try:
        train_ds = RecursiveBinaryDataset(TRAIN_DIR, transform=train_transform)
        val_ds = RecursiveBinaryDataset(VAL_DIR, transform=val_transform)
    except Exception as e:
        print(f"数据集错误: {e}")
        return

    train_loader = DataLoader(train_ds, batch_size=PHYSICAL_BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=PHYSICAL_BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=True)

    model = AIGCDetector(num_classes=2, embed_dim=96).to(DEVICE)

    # 加入 Weight Decay 防止过拟合
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    criterion = nn.CrossEntropyLoss()

    use_amp = torch.cuda.is_available()
    scaler = torch.amp.GradScaler('cuda') if use_amp else None

    best_acc = 0.0
    print(f"开始训练... (Data Augmentation V2)")

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        optimizer.zero_grad()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS}")

        for i, (inputs, labels) in enumerate(pbar):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            # 混合精度训练
            if use_amp:
                with torch.amp.autocast('cuda'):
                    outputs, recon_loss = model(inputs)
                    cls_loss = criterion(outputs, labels)
                    loss = (cls_loss + recon_loss) / ACCUM_STEPS

                scaler.scale(loss).backward()
            else:
                outputs, recon_loss = model(inputs)
                cls_loss = criterion(outputs, labels)
                loss = (cls_loss + recon_loss) / ACCUM_STEPS
                loss.backward()

            if (i + 1) % ACCUM_STEPS == 0:
                if use_amp:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()

            running_loss += loss.item() * ACCUM_STEPS
            if i % 50 == 0:
                pbar.set_postfix(loss=loss.item() * ACCUM_STEPS)

        scheduler.step()

        # 验证循环
        model.eval()
        all_targets = []
        all_probs = []

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                if use_amp:
                    with torch.amp.autocast('cuda'):
                        outputs, _ = model(inputs)
                else:
                    outputs, _ = model(inputs)

                probs = torch.softmax(outputs, dim=1)[:, 1]
                all_targets.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().float().numpy())

        all_preds = [1 if p > 0.5 else 0 for p in all_probs]
        val_acc = accuracy_score(all_targets, all_preds) * 100
        val_ap = average_precision_score(all_targets, all_probs) * 100

        avg_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch + 1} | Loss: {avg_loss:.4f} | Acc: {val_acc:.2f}% | AP: {val_ap:.2f}%")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "best_rswa_model_v3.pth")
            print(f"模型已保存: best_rswa_model_v3.pth")


if __name__ == "__main__":
    train_model()
