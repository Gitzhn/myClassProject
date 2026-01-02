import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import onnx
import onnxruntime as ort
from typing import Tuple

# ===================== 模型定义（与训练代码完全一致）=====================
class EmotionCNN(nn.Module):
    def __init__(self, num_classes=7):
        super(EmotionCNN, self).__init__()
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
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
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

# ===================== 数据集类（与训练代码完全一致）=====================
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
        # 转换为PyTorch张量：(H,W,C) → (C,H,W)
        image = torch.FloatTensor(image).permute(2, 0, 1)
        label = torch.LongTensor([label])[0]
        return image, label

# ===================== 数据处理函数（复用训练代码逻辑）=====================
def process_dataset_subset(sub_df):
    """处理单个数据集子集，与训练代码完全一致"""
    images = []
    labels = []
    for _, row in sub_df.iterrows():
        # 像素处理：字符串→48×48灰度图→归一化→扩展通道
        pixel_str = row['pixels']
        pixel_list = list(map(int, pixel_str.split()))
        img_array = np.array(pixel_list, dtype=np.float32).reshape(48, 48) / 255.0
        img_array = np.expand_dims(img_array, axis=-1)  # (48,48,1)
        images.append(img_array)
        labels.append(row['emotion'])
    return np.array(images), np.array(labels)

# ===================== 数据加载函数（按Usage字段划分，与训练代码一致）=====================
def load_all_datasets(csv_path: str) -> Tuple[
    Tuple[np.ndarray, np.ndarray],  # 训练集 (X_train, y_train)
    Tuple[np.ndarray, np.ndarray],  # 测试集 (X_test, y_test)
    Tuple[np.ndarray, np.ndarray]   # 全量数据 (X_all, y_all)
]:
    """
    按训练代码的划分逻辑加载数据：
    - 训练集：Usage=Training
    - 测试集：Usage=PublicTest
    - 全量数据：Training + PrivateTest + PublicTest
    """
    print(f"正在加载数据: {csv_path}")
    df = pd.read_csv(csv_path)

    # 1. 按Usage字段筛选（与训练代码完全一致）
    train_df = df[df['Usage'] == 'Training']       # 训练集
    test_df = df[df['Usage'] == 'PublicTest']      # 测试集（与训练代码测试集一致）
    all_df = df                                    # 全量数据

    # 2. 处理各子集数据
    X_train, y_train = process_dataset_subset(train_df)
    X_test, y_test = process_dataset_subset(test_df)
    X_all, y_all = process_dataset_subset(all_df)

    # 打印数据量统计
    print(f"\n数据集大小统计：")
    print(f"训练集（Usage=Training）: {len(X_train)} 张")
    print(f"测试集（Usage=PublicTest）: {len(X_test)} 张")
    print(f"全量数据: {len(X_all)} 张")

    return (X_train, y_train), (X_test, y_test), (X_all, y_all)

# ===================== 模型转换函数（保持原逻辑，适配训练模型路径）=====================
def convert_pth_to_onnx(pth_path: str, onnx_path: str, input_size: Tuple[int, int, int, int] = (1, 1, 48, 48)):
    print(f"\n正在加载 PyTorch 模型: {pth_path}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载模型（适配训练代码的保存格式）
    checkpoint = torch.load(pth_path, map_location=device)
    num_classes = checkpoint.get('config', {}).get('NUM_CLASSES', 7)
    model = EmotionCNN(num_classes=num_classes).to(device)
    
    # 加载权重
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()  # 评估模式（关闭dropout）
    print(f"模型已加载，类别数: {num_classes}")
    
    # 示例输入
    dummy_input = torch.randn(input_size).to(device)
    
    # 导出ONNX
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        },
        verbose=False
    )
    
    print(f"ONNX 模型已保存: {onnx_path}")
    
    # 验证ONNX模型
    print("正在验证 ONNX 模型...")
    onnx_model = onnx.load(onnx_path)
    try:
        onnx.checker.check_model(onnx_model)
        print("✓ ONNX 模型验证通过")
    except onnx.checker.ValidationError as e:
        print(f"✗ ONNX 模型验证失败: {e}")
        raise
    
    return model

# ===================== 准确率评估函数（支持多数据集评估）=====================
def evaluate_pytorch_model(model: nn.Module, data_loader: DataLoader, device: torch.device, dataset_name: str) -> float:
    """评估PyTorch模型在指定数据集上的准确率"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    accuracy = 100.0 * correct / total
    print(f"PyTorch 模型 - {dataset_name} 准确率: {accuracy:.4f}%")
    return accuracy

def evaluate_onnx_model(onnx_path: str, data_loader: DataLoader, dataset_name: str) -> float:
    """评估ONNX模型在指定数据集上的准确率"""
    ort_session = ort.InferenceSession(onnx_path)
    correct = 0
    total = 0
    
    for inputs, targets in data_loader:
        inputs_np = inputs.numpy()
        targets_np = targets.numpy()
        
        outputs = ort_session.run(None, {'input': inputs_np})
        predictions = np.argmax(outputs[0], axis=1)
        
        total += len(targets_np)
        correct += np.sum(predictions == targets_np)
    
    accuracy = 100.0 * correct / total
    print(f"ONNX 模型 - {dataset_name} 准确率:    {accuracy:.4f}%")
    return accuracy

# ===================== 主函数（新增多数据集评估逻辑）=====================
def main():
    # 配置路径（适配训练代码的模型保存路径）
    pth_model_path = "./emotion_cnn.pth"
    onnx_model_path = "./emotion_cnn.onnx"
    
    # 尝试多个CSV路径（与训练代码一致）
    csv_paths = [
        "./fer2013.csv/fer2013.csv",
        "./emotionModel/emotionModel/fer2013.csv",
        "./fer2013.csv"
    ]
    
    csv_path = None
    for path in csv_paths:
        if os.path.exists(path):
            csv_path = path
            break
    
    if csv_path is None:
        raise FileNotFoundError(f"找不到数据集文件，尝试过的路径: {csv_paths}")
    
    if not os.path.exists(pth_model_path):
        raise FileNotFoundError(f"找不到 PyTorch 模型文件: {pth_model_path}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    print("=" * 80)

    # 1. 转换模型（PyTorch → ONNX）
    print("步骤 1: 转换 PyTorch 模型为 ONNX")
    print("=" * 80)
    model = convert_pth_to_onnx(pth_model_path, onnx_model_path)
    print("=" * 80)

    # 2. 加载数据（按Usage划分，与训练代码一致）
    print("\n步骤 2: 加载数据（按Usage字段划分）")
    print("=" * 80)
    (X_train, y_train), (X_test, y_test), (X_all, y_all) = load_all_datasets(csv_path)
    
    # 创建三个数据集的数据加载器
    batch_size = 64
    train_dataset = FER2013Dataset(X_train, y_train)
    test_dataset = FER2013Dataset(X_test, y_test)
    all_dataset = FER2013Dataset(X_all, y_all)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    all_loader = DataLoader(all_dataset, batch_size=batch_size, shuffle=False)
    print("=" * 80)

    # 3. 评估 PyTorch 模型（训练集+测试集+全量数据）
    print("\n步骤 3: 评估 PyTorch 模型准确率")
    print("=" * 80)
    pytorch_train_acc = evaluate_pytorch_model(model, train_loader, device, "训练集（Usage=Training）")
    pytorch_test_acc = evaluate_pytorch_model(model, test_loader, device, "测试集（Usage=PublicTest）")
    pytorch_all_acc = evaluate_pytorch_model(model, all_loader, device, "全量数据")
    print("=" * 80)

    # 4. 评估 ONNX 模型（训练集+测试集+全量数据）
    print("\n步骤 4: 评估 ONNX 模型准确率")
    print("=" * 80)
    onnx_train_acc = evaluate_onnx_model(onnx_model_path, train_loader, "训练集（Usage=Training）")
    onnx_test_acc = evaluate_onnx_model(onnx_model_path, test_loader, "测试集（Usage=PublicTest）")
    onnx_all_acc = evaluate_onnx_model(onnx_model_path, all_loader, "全量数据")
    print("=" * 80)

    # 5. 对比结果汇总
    print("\n步骤 5: 准确率对比汇总")
    print("=" * 80)
    print(f"{'数据集':<25} {'PyTorch准确率':<15} {'ONNX准确率':<15} {'差异':<10}")
    print("-" * 80)
    train_diff = abs(pytorch_train_acc - onnx_train_acc)
    test_diff = abs(pytorch_test_acc - onnx_test_acc)
    all_diff = abs(pytorch_all_acc - onnx_all_acc)
    
    print(f"训练集（Usage=Training）{'':<5} {pytorch_train_acc:.4f}% {'':<5} {onnx_train_acc:.4f}% {'':<5} {train_diff:.6f}%")
    print(f"测试集（Usage=PublicTest）{'':<3} {pytorch_test_acc:.4f}% {'':<5} {onnx_test_acc:.4f}% {'':<5} {test_diff:.6f}%")
    print(f"全量数据{'':<15} {pytorch_all_acc:.4f}% {'':<5} {onnx_all_acc:.4f}% {'':<5} {all_diff:.6f}%")
    print("=" * 80)

    # 转换质量判断
    max_diff = max(train_diff, test_diff, all_diff)
    if max_diff < 0.01:
        print("✓ 转换成功！双模型准确率几乎完全一致（最大差异 < 0.01%）")
    elif max_diff < 0.1:
        print("⚠ 转换基本成功！存在微小数值差异（最大差异 < 0.1%），属正常现象")
    else:
        print("✗ 警告：双模型准确率差异较大（最大差异 ≥ 0.1%），请检查转换流程")
    
    print(f"\nONNX 模型已保存到: {onnx_model_path}")

if __name__ == "__main__":
    main()