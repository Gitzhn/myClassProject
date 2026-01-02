import os
import pandas as pd
import numpy as np
from PIL import Image

# ====================== 核心配置（修改这两个路径即可） ======================
csv_path = r"C:\Users\32252\Desktop\新建文件夹\fer2013.csv"  # 你的本地CSV文件路径
output_dir = r"C:\Users\32252\Desktop\新建文件夹\fer2013_images"  # 图片输出目录
# ===========================================================================

# 表情标签映射（方便分类保存，可选）
emotion_map = {
    0: "angry",
    1: "disgust",
    2: "fear",
    3: "happy",
    4: "sad",
    5: "surprise",
    6: "neutral"
}

# 创建分类目录（按 训练/测试 + 表情 划分，更易使用）
def create_dirs():
    for usage in ["train", "test"]:
        for label_name in emotion_map.values():
            dir_path = os.path.join(output_dir, usage, label_name)
            os.makedirs(dir_path, exist_ok=True)

# 转换CSV到图片
def convert_csv_to_images():
    # 读取本地CSV
    df = pd.read_csv(csv_path)
    total_samples = len(df)
    print(f"开始转换 {total_samples} 张图片...")

    for idx, row in df.iterrows():
        # 1. 提取核心信息
        emotion_idx = row["emotion"]  # 表情标签（0-6）
        pixels = row["pixels"]       # 像素字符串
        usage = row["Usage"]         # 数据划分（Training/PublicTest/PrivateTest）

        # 2. 合并PublicTest/PrivateTest为test，Training为train
        save_usage = "train" if usage == "Training" else "test"

        # 3. 像素字符串转48×48灰度数组
        pixel_array = np.array(pixels.split(), dtype=np.uint8).reshape(48, 48)

        # 4. 创建并保存图片
        img = Image.fromarray(pixel_array)  # 灰度图无需指定模式，PIL自动识别
        save_path = os.path.join(
            output_dir, save_usage, emotion_map[emotion_idx], f"img_{idx}.png"
        )
        img.save(save_path)

        # 5. 打印进度（每1000张更清晰）
        if idx % 1000 == 0:
            print(f"已完成 {idx}/{total_samples} 张")

    print(f"✅ 转换完成！所有图片保存在：{output_dir}")

# 主执行逻辑
if __name__ == "__main__":
    # 先创建目录，再转换
    create_dirs()
    convert_csv_to_images()