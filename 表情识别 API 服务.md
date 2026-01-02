# 表情识别 API 服务（昇腾 Atlas 200 DK）

## 功能说明

基于昇腾 Atlas 200 DK 硬件，提供表情识别 API 服务，支持上传图片识别 7 种情绪（生气、厌恶、恐惧、开心、中性、悲伤、惊讶），返回识别结果及置信度。

## 前提条件



1. 已部署昇腾 Atlas 200 DK 环境，配置好 ACL 相关环境变量

2. 安装依赖 Python 库：`flask numpy pillow`

3. 准备训练好的`pth`格式模型文件



### 1. 进入工作目录

连接Atlas 200 DK后，运行下面的命令，进入工作目录：
```
cd smartFacialRecognition/model
```

### 2. 转换 PTH 模型到 ONNX 格式
#### (若已有om模型，直接进入步骤5，启动api)

运行模型转换脚本，将 PyTorch 的`.pth`模型转为 ONNX 格式：
```
python3 pthToOnnx.py
```

### 3. 转换 ONNX 模型到 OM 格式（昇腾专用）

执行 Shell 脚本，将 ONNX 模型转为昇腾 NPU 支持的 OM 模型：

```
chmod +x onnxToOm.sh # 添加脚本执行权限
./onnxToOm.sh # 执行转换
```

### 4. （可选）验证 OM 模型准确率

若需验证转换后 OM 模型的识别准确率，执行验证脚本：

```
python3 validate_om_model.py
```

### 5. 启动 API 服务

启动 Flask API 服务，提供表情识别接口：

```
python3 api.py
```

服务启动后，默认监听地址：`http://192.168.137.2:5000`

## API 调用示例

### 终端 curl 调用

将`./test.jpg`替换为你的本地图片路径，执行以下命令：


```
curl -X POST http://192.168.137.2:5000/api/v1/predict -F "image=@./test.jpg"
```

### 返回结果示例



```
{
    "status": "success",
    "data": {
        "predicted_emotion": "happy",
        "confidence": 0.95,
        "emotions_distribution": [
            {"emotion": "neutral", "confidence": 0.02},
            {"emotion": "angry", "confidence": 0.01},
            {"emotion": "disgust", "confidence": 0.00},
            {"emotion": "fear", "confidence": 0.01},
            {"emotion": "happy", "confidence": 0.95},
            {"emotion": "sad", "confidence": 0.01},
            {"emotion": "surprise", "confidence": 0.00}
        ],
        "processing_time": 0.15,
        "timestamp": "2024-01-01T10:00:00Z"
    }
}
```

## 注意事项



1. 确保`pthToOnnx.py`、`onnxToOm.sh`脚本中模型路径配置正确

2. 运行 API 服务前，需先配置昇腾环境变量（`ASCEND_HOME`、`LD_LIBRARY_PATH`）

3. 图片支持 JPG/PNG 格式，建议单张图片大小不超过 5MB

4. 若端口 5000 被占用，可修改`emotion_api.py`中`app.run(port=xxx)`指定其他端口
