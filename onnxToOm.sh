#!/bin/bash
# Atlas 200 DK - ONNX 转 OM 模型转换脚本
# 用于将 emotion_cnn.onnx 转换为 emotion_cnn.om

# 模型路径配置
ONNX_MODEL="./emotion_cnn.onnx"
OM_MODEL="./emotion_cnn.om"

# 检查 ONNX 模型是否存在
if [ ! -f "$ONNX_MODEL" ]; then
    echo "错误: 找不到 ONNX 模型文件: $ONNX_MODEL"
    exit 1
fi

echo "=========================================="
echo "开始转换 ONNX 模型为 OM 模型"
echo "=========================================="
echo "输入模型: $ONNX_MODEL"
echo "输出模型: $OM_MODEL"
echo ""

# ATC 转换命令
# 参数说明：
# --model: ONNX 模型路径
# --framework=5: 表示 ONNX 框架（0: Caffe, 1: MindSpore, 3: TensorFlow, 5: ONNX）
# --output: 输出 OM 模型路径（不含扩展名）
# --soc_version: 昇腾芯片版本（根据实际设备选择）
#   - Ascend310: 用于 Atlas 200 DK
#   - Ascend310P3: 用于 Atlas 200 DK (如果支持)
# --input_shape: 输入形状 (batch, channels, height, width)
# --input_format: 输入格式 NCHW
# --precision_mode: 精度模式
#   - force_fp32: 强制 FP32，最无损，但速度较慢
#   - allow_fp32_to_fp16: 允许 FP32 转 FP16，可能有精度损失但速度快
#   - allow_mix_precision: 混合精度
# --op_select_implmode: 算子选择模式
#   - high_precision: 高精度实现，更无损
#   - high_performance: 高性能实现，可能略有精度损失
# --log: 日志级别

# 方案1: 最高精度模式（推荐，最无损）
echo "使用最高精度模式转换（推荐）..."
atc --model=$ONNX_MODEL \
    --framework=5 \
    --output=${OM_MODEL%.om} \
    --soc_version=Ascend310 \
    --input_shape="input:1,1,48,48" \
    --input_format=NCHW \
    --precision_mode=force_fp32 \
    --op_select_implmode=high_precision \
    --log=info

# 如果上面的命令失败，可以尝试以下备选方案：

# 方案2: 如果设备是 Ascend310P3，使用以下命令
# atc --model=$ONNX_MODEL \
#     --framework=5 \
#     --output=${OM_MODEL%.om} \
#     --soc_version=Ascend310P3 \
#     --input_shape="input:1,1,48,48" \
#     --input_format=NCHW \
#     --precision_mode=force_fp32 \
#     --op_select_implmode=high_precision \
#     --log=info

# 方案3: 如果 force_fp32 不支持，使用混合精度（仍保持较高精度）
# atc --model=$ONNX_MODEL \
#     --framework=5 \
#     --output=${OM_MODEL%.om} \
#     --soc_version=Ascend310 \
#     --input_shape="input:1,1,48,48" \
#     --input_format=NCHW \
#     --precision_mode=allow_mix_precision \
#     --op_select_implmode=high_precision \
#     --log=info

# 检查转换结果
if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "转换成功！"
    echo "=========================================="
    echo "OM 模型已保存到: $OM_MODEL"
    ls -lh $OM_MODEL
else
    echo ""
    echo "=========================================="
    echo "转换失败！"
    echo "=========================================="
    echo "请检查错误信息，可能需要："
    echo "1. 确认 soc_version 是否正确（使用 'npu-smi info' 查看）"
    echo "2. 确认 ONNX 模型格式是否正确"
    echo "3. 尝试使用备选方案（注释中的方案2或方案3）"
    exit 1
fi

