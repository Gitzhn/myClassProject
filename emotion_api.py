import os
import sys
import ctypes
import traceback
import numpy as np
import time
from io import BytesIO
from PIL import Image, ImageOps
from flask import Flask, request, jsonify
import datetime
import logging

# 配置日志（打印推理过程）
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 初始化Flask应用
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024  # 限制上传大小5MB

# 昇腾ACL初始化（对齐验证代码的常量定义）
try:
    import acl
    # 手动定义常量（和验证代码一致）
    ACL_MEM_MALLOC_HUGE_FIRST = 0
    MEMCPY_HOST_TO_DEVICE = 1
    MEMCPY_DEVICE_TO_HOST = 2
except ImportError:
    logger.error("错误：未找到ACL模块，请配置昇腾环境")
    sys.exit(1)

# ===================== 核心配置（与验证代码对齐）=====================
CONFIG = {
    'MODEL_PATH': './emotion_cnn.om',          # OM模型路径（和验证代码一致）
    'DEVICE_ID': 0,                            # NPU设备ID
    'NUM_CLASSES': 7,                          # 情绪类别数（0-6）
    # 预处理配置（和验证代码完全一致）
    'PREPROCESS': {
        'mean': 0.0,       # 训练时的均值
        'std': 1.0,        # 训练时的标准差
        'scale': 1.0/255.0 # 像素归一化系数
    },
    'INPUT_SHAPE': (1, 1, 48, 48),             # 模型输入形状 (N, C, H, W)
    'EXPECT_INPUT_SIZE': 48*48*1*4             # 预期输入大小（和验证代码一致）
}

# 情绪标签映射（调整为：0=Angry,1=Disgust,2=Fear,3=Happy,4=Sad,5=Surprise,6=Neutral）
EMOTION_LABELS = {
    0: "angry",        # 生气（0=Angry）
    1: "disgust",      # 厌恶（1=Disgust）
    2: "fear",         # 恐惧（2=Fear）
    3: "happy",        # 开心（3=Happy）
    4: "sad",          # 悲伤（4=Sad）
    5: "surprise",     # 惊讶（5=Surprise）
    6: "neutral"       # 中性（6=Neutral）
}

# 输出情绪分布顺序
EMOTION_ORDER = ["neutral", "angry", "disgust", "fear", "happy", "sad", "surprise"]

# 全局资源（服务启动时初始化，所有请求复用）
g_global_resources = {
    'context': None,
    'model_id': None,
    'model_desc': None,
    'input_buf': None,
    'output_buf': None,
    'input_size': 0,
    'output_size': 0,
    'initialized': False  # 标记是否已初始化
}


def stable_softmax(x):
    """数值稳定的Softmax计算（和验证代码一致）"""
    x_max = np.max(x)
    exp_x = np.exp(x - x_max)
    sum_exp = np.sum(exp_x)
    return exp_x / sum_exp if sum_exp != 0 else np.ones_like(x) / len(x)


def align_train_preprocess(image):
    """预处理逻辑（和验证代码完全对齐）"""
    try:
        # 1. 转换为灰度图
        gray_img = ImageOps.grayscale(image)
        # 2. Resize为48x48
        resized_img = gray_img.resize((48, 48), Image.LANCZOS)
        if resized_img.size != (48, 48):
            raise ValueError(f"Resize后尺寸错误：{resized_img.size}（预期48x48）")
        # 3. 转换为numpy数组（0-255 uint8）
        pixel_array = np.array(resized_img, dtype=np.uint8)
        if pixel_array.shape != (48, 48):
            raise ValueError(f"像素数组形状错误：{pixel_array.shape}（预期48x48）")
        # 4. 归一化（和验证代码一致）
        img = pixel_array.astype(np.float32) * CONFIG['PREPROCESS']['scale']
        img = (img - CONFIG['PREPROCESS']['mean']) / CONFIG['PREPROCESS']['std']
        # 5. 调整维度为NCHW格式
        img = np.expand_dims(img, axis=0)  # batch维度
        img = np.expand_dims(img, axis=0)  # 通道维度
        # 校验
        if img.shape != CONFIG['INPUT_SHAPE'] or img.nbytes != CONFIG['EXPECT_INPUT_SIZE']:
            raise ValueError(f"预处理后数据格式不匹配：形状{img.shape}，字节数{img.nbytes}")
        logger.debug(f"预处理完成：形状{img.shape}，字节数{img.nbytes}")
        return img
    except Exception as e:
        logger.error(f"预处理失败：{str(e)}")
        raise


def global_acl_init():
    """【服务启动时仅执行一次】初始化全局ACL和模型资源"""
    if g_global_resources['initialized']:
        return True
    try:
        # 1. 检查模型文件（绝对路径验证）
        model_path = os.path.abspath(CONFIG['MODEL_PATH'])
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"OM模型不存在：{model_path}")
        if not os.access(model_path, os.R_OK):
            raise PermissionError(f"无模型读取权限：{model_path}")
        logger.info(f"验证模型文件：{model_path}（存在且有权限）")

        # 2. 初始化ACL（仅一次）
        ret = acl.init()
        ret = ret[-1] if isinstance(ret, (tuple, list)) else ret
        if ret != 0:
            raise RuntimeError(f"ACL初始化失败，错误码：{ret}")

        # 3. 设置NPU设备（仅一次）
        ret = acl.rt.set_device(CONFIG['DEVICE_ID'])
        ret = ret[-1] if isinstance(ret, (tuple, list)) else ret
        if ret != 0:
            raise RuntimeError(f"设置NPU设备失败，错误码：{ret}")

        # 4. 创建设备上下文（仅一次）
        create_ret = acl.rt.create_context(CONFIG['DEVICE_ID'])
        if isinstance(create_ret, (tuple, list)):
            g_global_resources['context'], ret = create_ret
        else:
            g_global_resources['context'] = create_ret
            ret = 0
        if ret != 0 or not g_global_resources['context']:
            raise RuntimeError(f"创建上下文失败，错误码：{ret}")

        # 5. 加载OM模型（仅一次）
        load_ret = acl.mdl.load_from_file(model_path)
        if isinstance(load_ret, (tuple, list)):
            g_global_resources['model_id'], ret = load_ret
        else:
            g_global_resources['model_id'] = load_ret
            ret = 0
        if ret != 0 or not g_global_resources['model_id']:
            raise RuntimeError(f"加载OM模型失败，错误码：{ret}")

        # 6. 获取模型描述和输入输出大小（仅一次）
        g_global_resources['model_desc'] = acl.mdl.create_desc()
        acl.mdl.get_desc(g_global_resources['model_desc'], g_global_resources['model_id'])
        g_global_resources['input_size'] = acl.mdl.get_input_size_by_index(g_global_resources['model_desc'], 0)
        g_global_resources['output_size'] = acl.mdl.get_output_size_by_index(g_global_resources['model_desc'], 0)

        # 验证输入大小（和验证代码一致）
        logger.info(f"模型输入大小验证：")
        logger.info(f"  模型要求：{g_global_resources['input_size']} 字节")
        logger.info(f"  预期大小：{CONFIG['EXPECT_INPUT_SIZE']} 字节")
        if g_global_resources['input_size'] != CONFIG['EXPECT_INPUT_SIZE']:
            logger.warning("输入大小不匹配，使用预期大小")
            g_global_resources['input_size'] = CONFIG['EXPECT_INPUT_SIZE']

        # 7. 申请NPU设备内存（仅一次）
        malloc_ret = acl.rt.malloc(int(g_global_resources['input_size']), int(ACL_MEM_MALLOC_HUGE_FIRST))
        if isinstance(malloc_ret, (tuple, list)):
            g_global_resources['input_buf'], ret = malloc_ret
        else:
            g_global_resources['input_buf'] = malloc_ret
            ret = 0
        if ret != 0 or not g_global_resources['input_buf']:
            raise RuntimeError(f"申请输入内存失败，错误码：{ret}")

        malloc_ret = acl.rt.malloc(int(g_global_resources['output_size']), int(ACL_MEM_MALLOC_HUGE_FIRST))
        if isinstance(malloc_ret, (tuple, list)):
            g_global_resources['output_buf'], ret = malloc_ret
        else:
            g_global_resources['output_buf'] = malloc_ret
            ret = 0
        if ret != 0 or not g_global_resources['output_buf']:
            raise RuntimeError(f"申请输出内存失败，错误码：{ret}")

        g_global_resources['initialized'] = True
        logger.info("✅ 全局ACL+模型初始化成功（服务启动仅一次）")
        return True
    except Exception as e:
        logger.error(f"❌ 全局ACL初始化失败：{str(e)}")
        global_acl_cleanup()
        raise


def om_model_infer(img_tensor):
    """【每次请求仅执行】模型推理（复用全局资源，仅处理图片）"""
    try:
        # 1. 转换为字节流（和验证代码一致）
        img_bytes = img_tensor.tobytes()
        actual_size = len(img_bytes)
        if actual_size > g_global_resources['input_size']:
            img_bytes = img_bytes[:g_global_resources['input_size']]
            logger.warning(f"输入数据过长，截断至{g_global_resources['input_size']}字节")
        elif actual_size < g_global_resources['input_size']:
            logger.warning(f"输入数据过短，补零至{g_global_resources['input_size']}字节")
            img_bytes = img_bytes.ljust(g_global_resources['input_size'], b'\x00')

        # 2. 创建内存缓冲区（局部变量，推理后自动释放）
        img_buffer = ctypes.create_string_buffer(img_bytes)
        src_ptr = ctypes.addressof(img_buffer)

        # 3. 主机→NPU内存拷贝（复用全局输入内存）
        memcpy_ret = acl.rt.memcpy(
            int(g_global_resources['input_buf']),
            int(g_global_resources['input_size']),
            int(src_ptr),
            int(len(img_bytes)),
            int(MEMCPY_HOST_TO_DEVICE)
        )
        ret = memcpy_ret[-1] if isinstance(memcpy_ret, (tuple, list)) else memcpy_ret
        if ret != 0:
            raise RuntimeError(f"内存拷贝失败，错误码：{ret}")

        # 4. 创建数据集和缓冲区（局部变量，推理后释放）
        dataset_in = acl.mdl.create_dataset()
        dataset_out = acl.mdl.create_dataset()

        # 输入缓冲区（局部）
        buf_ret = acl.create_data_buffer(int(g_global_resources['input_buf']), int(len(img_bytes)))
        input_buf_obj = buf_ret[0] if isinstance(buf_ret, (tuple, list)) else buf_ret
        if not input_buf_obj:
            raise RuntimeError("创建输入缓冲区失败")
        add_ret = acl.mdl.add_dataset_buffer(dataset_in, input_buf_obj)
        ret = add_ret[-1] if isinstance(add_ret, (tuple, list)) else add_ret
        if ret != 0:
            raise RuntimeError(f"添加输入缓冲区失败，错误码：{ret}")

        # 输出缓冲区（局部，复用全局输出内存）
        buf_ret = acl.create_data_buffer(int(g_global_resources['output_buf']), int(g_global_resources['output_size']))
        output_buf_obj = buf_ret[0] if isinstance(buf_ret, (tuple, list)) else buf_ret
        if not output_buf_obj:
            raise RuntimeError("创建输出缓冲区失败")
        add_ret = acl.mdl.add_dataset_buffer(dataset_out, output_buf_obj)
        ret = add_ret[-1] if isinstance(add_ret, (tuple, list)) else ret
        if ret != 0:
            raise RuntimeError(f"添加输出缓冲区失败，错误码：{ret}")

        # 5. 执行推理（复用全局模型）
        exec_ret = acl.mdl.execute(g_global_resources['model_id'], dataset_in, dataset_out)
        ret = exec_ret[-1] if isinstance(exec_ret, (tuple, list)) else ret
        if ret != 0:
            raise RuntimeError(f"推理执行失败，错误码：{ret}")

        # 6. NPU→主机内存拷贝
        out_buffer = ctypes.create_string_buffer(g_global_resources['output_size'])
        dst_ptr = ctypes.addressof(out_buffer)
        memcpy_ret = acl.rt.memcpy(
            int(dst_ptr),
            int(g_global_resources['output_size']),
            int(g_global_resources['output_buf']),
            int(g_global_resources['output_size']),
            int(MEMCPY_DEVICE_TO_HOST)
        )
        ret = memcpy_ret[-1] if isinstance(memcpy_ret, (tuple, list)) else ret
        if ret != 0:
            raise RuntimeError(f"输出拷贝失败，错误码：{ret}")

        # 7. 解析结果
        output_data = np.frombuffer(out_buffer.raw, dtype=np.float32)[:CONFIG['NUM_CLASSES']]
        pred_probs = stable_softmax(output_data)
        pred_label = np.argmax(pred_probs).item()
        max_confidence = pred_probs[pred_label]  # 获取最高置信度
        emotion_name = EMOTION_LABELS[pred_label]  # 获取情绪名称

        # ========== 新增：打印推理结果（仅最高置信度） ==========
        logger.info(f"【推理结果】情绪：{emotion_name}，最高置信度：{max_confidence:.4f}")

        # 8. 释放本次请求的临时资源（关键：仅释放局部资源，全局资源保留）
        acl.destroy_data_buffer(input_buf_obj)
        acl.destroy_data_buffer(output_buf_obj)
        acl.mdl.destroy_dataset(dataset_in)
        acl.mdl.destroy_dataset(dataset_out)
        
        # 9. 释放/覆盖图片相关内存（核心：仅释放图片内存）
        del img_tensor  # 删除张量
        if 'img_bytes' in locals():
            del img_bytes  # 条件删除字节流
        del img_buffer, out_buffer  # 删除缓冲区
        import gc
        gc.collect()  # 强制垃圾回收

        logger.debug(f"推理成功：标签={pred_label}（{EMOTION_LABELS[pred_label]}）")
        return pred_label, pred_probs
    except Exception as e:
        logger.error(f"推理失败：{str(e)}")
        # 异常释放临时资源
        try:
            if 'dataset_in' in locals():
                acl.mdl.destroy_dataset(dataset_in)
            if 'dataset_out' in locals():
                acl.mdl.destroy_dataset(dataset_out)
        except:
            pass
        # 异常时也释放图片内存
        if 'img_tensor' in locals():
            del img_tensor
        if 'img_bytes' in locals():
            del img_bytes
        gc.collect()
        return -1, None


def global_acl_cleanup():
    """【服务退出时仅执行一次】释放全局资源"""
    if not g_global_resources['initialized']:
        return
    try:
        # 释放内存
        if g_global_resources.get('output_buf'):
            acl.rt.free(int(g_global_resources['output_buf']))
        if g_global_resources.get('input_buf'):
            acl.rt.free(int(g_global_resources['input_buf']))
        # 释放模型
        if g_global_resources.get('model_desc'):
            acl.mdl.destroy_desc(g_global_resources['model_desc'])
        if g_global_resources.get('model_id'):
            unload_ret = acl.mdl.unload(int(g_global_resources['model_id']))
            ret = unload_ret[-1] if isinstance(unload_ret, (tuple, list)) else ret
            if ret != 0:
                logger.warning(f"卸载模型警告，错误码：{ret}")
        # 释放上下文
        if g_global_resources.get('context'):
            acl.rt.destroy_context(g_global_resources['context'])
        # 重置设备+终结ACL
        acl.rt.reset_device(CONFIG['DEVICE_ID'])
        acl.finalize()
        
        g_global_resources['initialized'] = False
        logger.info("✅ 全局ACL资源释放完成（服务退出仅一次）")
    except Exception as e:
        logger.warning(f"⚠️  全局资源释放警告：{str(e)}")


@app.route('/api/v1/predict', methods=['POST'])
def predict():
    """
    表情识别API接口（核心：复用全局资源，仅处理图片）
    接收：multipart/form-data格式的图片（key为image）
    返回：JSON格式识别结果
    """
    start_time = time.time()
    try:
        # 1. 检查上传参数
        if 'image' not in request.files:
            return jsonify({
                "status": "error",
                "message": "缺少image参数（请上传图片）"
            }), 400

        image_file = request.files['image']
        image_filename = image_file.filename  # 获取图片文件名
        if image_filename == '':
            return jsonify({
                "status": "error",
                "message": "未选择图片文件"
            }), 400

        # 2. 读取图片（局部变量，推理后释放）
        image = Image.open(image_file.stream)
        logger.info(f"接收图片：{image_filename}，格式：{image.format}，尺寸：{image.size}")

        # 3. 预处理（局部变量）
        img_tensor = align_train_preprocess(image)
        
        # 4. 关闭图片文件句柄（立即释放图片内存）
        image.close()
        del image  # 手动删除图片对象

        # 5. 模型推理（复用全局资源）
        pred_label, pred_probs = om_model_infer(img_tensor)
        if pred_label == -1 or pred_probs is None:
            raise RuntimeError("模型推理失败")

        # 6. 整理结果
        predicted_emotion = EMOTION_LABELS[pred_label]
        confidence = round(float(pred_probs[pred_label]), 4)

        # ========== 新增：打印本次请求的最终识别结果 ==========
        logger.info(f"【最终识别结果】图片：{image_filename} → 情绪：{predicted_emotion}，最高置信度：{confidence}")

        # 构建情绪分布
        emotions_distribution = []
        for emotion in EMOTION_ORDER:
            label = next(k for k, v in EMOTION_LABELS.items() if v == emotion)
            emotions_distribution.append({
                "emotion": emotion,
                "confidence": round(float(pred_probs[label]), 4)
            })

        # 7. 返回响应
        return jsonify({
            "status": "success",
            "data": {
                "predicted_emotion": predicted_emotion,
                "confidence": confidence,
                "emotions_distribution": emotions_distribution,
                "processing_time": round(time.time() - start_time, 4),
                "timestamp": datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
            }
        })

    except Exception as e:
        logger.error(f"请求处理失败：{str(e)}", exc_info=True)
        return jsonify({
            "status": "error",
            "message": f"服务器内部错误：{str(e)}"
        }), 500

    finally:
        # 最终保障：释放所有图片相关局部变量
        import gc
        gc.collect()


if __name__ == '__main__':
    """主函数：服务启动→全局初始化→启动Flask→退出释放"""
    logger.info("=====================================")
    logger.info("  表情识别API服务（昇腾Atlas 200 DK）  ")
    logger.info("  模式：全局初始化→请求仅处理图片→退出释放  ")
    logger.info("=====================================")

    # 1. 服务启动时初始化全局ACL+模型（仅一次）
    if not global_acl_init():
        sys.exit(1)

    # 2. 启动Flask服务（关闭多线程，避免资源冲突）
    try:
        port = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
        logger.info(f"API服务启动，监听地址：0.0.0.0:{port}")
        app.run(host='0.0.0.0', port=port, threaded=False)
    except KeyboardInterrupt:
        logger.info("\n🛑 接收到退出信号，释放全局资源...")
    finally:
        # 3. 服务退出时释放全局资源（仅一次）
        global_acl_cleanup()
        logger.info("🎉 API服务正常退出！")
