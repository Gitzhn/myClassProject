import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import json

# 设置matplotlib中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ===================== 配置参数 =====================
class Config:
    # 数据集路径：相对于本脚本文件（train/model）的位置，避免与当前工作目录依赖
    DATASET_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fer2013.csv")
    IMG_SIZE = (48, 48)             # 图像尺寸
    BATCH_SIZE = 64                 # 批次大小
    EPOCHS = 50                     # 训练轮数
    NUM_CLASSES = 7                 # 情绪类别数
    LEARNING_RATE = 0.001           # 学习率
    # 标签映射
    EMOTION_LABELS = {
        0: "生气", 1: "厌恶", 2: "恐惧", 3: "开心",
        4: "悲伤", 5: "惊讶", 6: "中性"
    }
    # 将模型保存目录放在当前脚本所在的 model 目录下，确保生成的 emotion_cnn.pth 存放在 train/model/saved_models_pytorch
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL_DIR = os.path.join(BASE_DIR, "saved_models_pytorch")  # 模型保存目录

config = Config()
os.makedirs(config.MODEL_DIR, exist_ok=True)  # 创建模型保存目录（不存在则创建）

# ===================== 数据集类 =====================
class FER2013Dataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        # 转换为PyTorch张量：(H,W,C) → (C,H,W)（PyTorch要求通道在前）
        image = torch.FloatTensor(image).permute(2, 0, 1)
        label = torch.LongTensor([label])[0]  # 标签转换为长整型张量
        return image, label

# ===================== CNN模型 =====================
class EmotionCNN(nn.Module):
    def __init__(self, num_classes=7):
        super(EmotionCNN, self).__init__()
        # 卷积层：4层卷积+池化+批归一化+dropout
        self.conv_layers = nn.Sequential(
            # 第一层
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),
            # 第二层
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),
            # 第三层
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),
            # 第四层
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),
        )
        # 全连接层：3层全连接+批归一化+dropout
        self.fc_layers = nn.Sequential(
            nn.Flatten(),  # 展平卷积输出（256*3*3）
            nn.Linear(256 * 3 * 3, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

# ===================== 辅助函数：数据处理 =====================
def process_dataset_subset(sub_df):
    """处理单个数据集子集（训练/验证/测试），转换为图像数组和标签数组"""
    images = []
    labels = []
    for _, row in sub_df.iterrows():
        # 处理像素数据：字符串→整数列表→48×48灰度图→归一化
        pixel_str = row['pixels']
        pixel_list = list(map(int, pixel_str.split()))
        img_array = np.array(pixel_list, dtype=np.float32).reshape(48, 48) / 255.0
        img_array = np.expand_dims(img_array, axis=-1)  # 增加通道维度（48,48,1）
        images.append(img_array)
        # 处理标签
        labels.append(row['emotion'])
    return np.array(images), np.array(labels)

# ===================== 辅助函数：训练过程可视化 =====================
def visualize_training(train_losses, val_losses, train_accs, val_accs):
    """可视化训练损失和准确率曲线"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    # 损失曲线
    axes[0].plot(train_losses, label='训练损失', color='#1f77b4')
    axes[0].plot(val_losses, label='验证损失', color='#ff7f0e')
    axes[0].set_xlabel('Epoch（轮数）')
    axes[0].set_ylabel('Loss（损失值）')
    axes[0].set_title('训练/验证损失曲线')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    # 准确率曲线
    axes[1].plot(train_accs, label='训练准确率', color='#1f77b4')
    axes[1].plot(val_accs, label='验证准确率', color='#ff7f0e')
    axes[1].set_xlabel('Epoch（轮数）')
    axes[1].set_ylabel('Accuracy（准确率 %）')
    axes[1].set_title('训练/验证准确率曲线')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    # 保存图片
    plt.tight_layout()
    plt.savefig(os.path.join(config.MODEL_DIR, 'training_curves.png'), dpi=300, bbox_inches='tight')
    plt.show()

# ===================== 辅助函数：混淆矩阵可视化 =====================
def visualize_confusion_matrix(y_true, y_pred):
    """可视化混淆矩阵并保存分类报告"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=list(config.EMOTION_LABELS.values()),
                yticklabels=list(config.EMOTION_LABELS.values()))
    plt.title('混淆矩阵（测试集）')
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.tight_layout()
    plt.savefig(os.path.join(config.MODEL_DIR, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
    plt.show()

    # 生成分类报告（精确率、召回率、F1分数）
    report = classification_report(
        y_true, y_pred,
        target_names=list(config.EMOTION_LABELS.values()),
        output_dict=True
    )
    report_path = os.path.join(config.MODEL_DIR, 'classification_report.json')
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n分类报告已保存至：{report_path}")

# ===================== 核心训练函数 =====================
def train_pytorch_model():
    """核心训练逻辑：加载数据→按Usage划分→训练→评估→保存模型"""
    print("="*50)
    print("使用PyTorch训练情绪识别CNN模型（按Usage字段划分数据）")
    print("="*50)
    # 打印环境信息
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"训练设备: {device}")
    print("="*50)

    # 1. 加载数据并按Usage字段筛选
    print("\n1. 加载数据集并按Usage划分...")
    df = pd.read_csv(config.DATASET_PATH)
    
    # 严格按Usage字段筛选：训练集=Training，验证集=PrivateTest，测试集使用文件中的所有数据
    train_df = df[df['Usage'] == 'Training']       # 训练集：仅使用Training数据
    val_df = df[df['Usage'] == 'PrivateTest']      # 验证集：使用PrivateTest数据
    test_df = df.copy()                             # 测试集：使用文件中的所有数据
    
    # 若需测试集同时包含PublicTest和PrivateTest，替换上面一行为：
    # test_df = df[df['Usage'].isin(['PublicTest', 'PrivateTest'])]
    
    # 处理每个子集的数据
    X_train, y_train = process_dataset_subset(train_df)
    X_val, y_val = process_dataset_subset(val_df)
    X_test, y_test = process_dataset_subset(test_df)
    
    # 打印数据集大小（符合FER2013原生划分）
    print(f"训练集大小: {len(X_train)} 张图片（Usage=Training）")
    print(f"验证集大小: {len(X_val)} 张图片（Usage=PrivateTest）")
    print(f"测试集大小: {len(X_test)} 张图片（使用文件中的所有数据）")

    # 2. 创建数据加载器（批量加载数据）
    print("\n2. 创建数据加载器...")
    train_dataset = FER2013Dataset(X_train, y_train)
    val_dataset = FER2013Dataset(X_val, y_val)
    test_dataset = FER2013Dataset(X_test, y_test)
    
    # 训练集开启打乱，验证集/测试集不打乱（适配CPU训练，稳定性更高）
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

    # 3. 初始化模型、损失函数、优化器
    print("\n3. 初始化模型和训练组件...")
    model = EmotionCNN(num_classes=config.NUM_CLASSES).to(device)  # 模型移到GPU/CPU
    criterion = nn.CrossEntropyLoss()  # 分类任务损失函数
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)  # 优化器
    # 学习率调度器（适配旧版本PyTorch，移除verbose参数）
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)

    # 4. 开始训练（记录损失和准确率）
    print("\n4. 开始训练（仅使用Training数据）...")
    train_losses = []  # 记录每轮训练损失
    val_losses = []    # 记录每轮验证损失
    train_accs = []    # 记录每轮训练准确率
    val_accs = []      # 记录每轮验证准确率

    for epoch in range(config.EPOCHS):
        # ---------------------- 训练阶段（仅用训练集数据）----------------------
        model.train()  # 训练模式（启用dropout、批归一化更新）
        running_train_loss = 0.0

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            # 数据移到设备
            inputs, targets = inputs.to(device), targets.to(device)
            # 前向传播
            optimizer.zero_grad()  # 清空梯度
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            # 反向传播+参数更新
            loss.backward()
            optimizer.step()
            # 统计训练损失（保留原始训练损失计算）
            running_train_loss += loss.item()

        # 计算本轮训练损失（与之前一致）
        train_loss = running_train_loss / len(train_loader)
        train_losses.append(train_loss)

        # 使用当前模型在完整训练集上评估训练准确率（符合“用当时训练的模型计算train_df准确率”）
        model.eval()
        train_correct = 0
        train_total = 0
        with torch.no_grad():
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                train_total += targets.size(0)
                train_correct += predicted.eq(targets).sum().item()
        train_acc = 100. * train_correct / train_total
        train_accs.append(train_acc)

        # ---------------------- 验证损失（仍使用 PrivateTest）----------------------
        running_val_loss = 0.0
        with torch.no_grad():  # 仍用 val_loader 计算验证损失（保持损失曲线不变）
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                running_val_loss += loss.item()
        val_loss = running_val_loss / len(val_loader)
        val_losses.append(val_loss)

        # 使用当前模型在测试集（test_df）上评估验证准确率（按你的要求）
        test_correct_for_val = 0
        test_total_for_val = 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                test_total_for_val += targets.size(0)
                test_correct_for_val += predicted.eq(targets).sum().item()
        val_acc = 100. * test_correct_for_val / test_total_for_val
        val_accs.append(val_acc)

        # 学习率调度（基于验证损失）
        scheduler.step(val_loss)

        # 每5轮打印一次训练信息
        if (epoch + 1) % 5 == 0:
            print(f"\nEpoch [{epoch + 1}/{config.EPOCHS}]")
            print(f"训练损失: {train_loss:.4f} | 训练准确率: {train_acc:.2f}%")
            print(f"验证损失: {val_loss:.4f} | 验证准确率: {val_acc:.2f}%")

    print("\n" + "="*50)
    print("训练完成！（训练仅使用Usage=Training的数据）")
    print("="*50)

    # 5. 测试集评估（用PublicTest数据）
    print("\n5. 测试集评估...")
    model.eval()
    test_correct = 0
    test_total = 0
    all_preds = []  # 记录所有预测标签
    all_targets = []  # 记录所有真实标签

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            test_total += targets.size(0)
            test_correct += predicted.eq(targets).sum().item()
            # 收集预测结果（用于混淆矩阵）
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    test_acc = 100. * test_correct / test_total
    print(f"\n测试集最终准确率: {test_acc:.2f}%")

    # 6. 保存模型（包含模型参数、配置、测试准确率）
    print("\n6. 保存模型...")
    model_path = os.path.join(config.MODEL_DIR, 'emotion_cnn.pth')
    torch.save({
        'model_state_dict': model.state_dict(),  # 模型参数
        'config': config.__dict__,               # 训练配置
        'test_accuracy': test_acc,               # 测试准确率
        'train_losses': train_losses,            # 训练损失记录
        'train_accs': train_accs                 # 训练准确率记录
    }, model_path)
    print(f"模型已保存至: {model_path}")

    # 7. 可视化结果（损失曲线、混淆矩阵）
    print("\n7. 生成可视化结果...")
    visualize_training(train_losses, val_losses, train_accs, val_accs)
    visualize_confusion_matrix(all_targets, all_preds)

    # 返回关键结果
    return model, train_accs, test_acc

# ===================== 主函数 =====================
if __name__ == "__main__":
    # 安装依赖提示（若未安装）
    print("提示：若未安装依赖，请先执行以下命令：")
    print("pip install torch torchvision pandas numpy scikit-learn matplotlib seaborn")
    print("="*50 + "\n")
    
    # 启动训练
    model, train_accuracies, test_accuracy = train_pytorch_model()
    
    # 最终输出关键信息
    print("\n" + "="*50)
    print("训练结果汇总：")
    print(f"最后一轮训练准确率: {train_accuracies[-1]:.2f}%")
    print(f"测试集最终准确率: {test_accuracy:.2f}%")
    print(f"模型文件路径: {os.path.join(config.MODEL_DIR, 'emotion_cnn.pth')}")
    print("="*50)