"""
OM模型全量推理验证（昇腾Atlas 200 DK）
核心：预处理严格对齐训练逻辑 + FP32精度推理 + 全量样本统计
支持训练集、测试集和全量数据的准确率评估，最终汇总展示
"""
import os
import sys
import ctypes
import traceback
import numpy as np
import time

# 强制适配昇腾21.0.3.1版本ACL
try:
    import acl
    # 手动定义常量
    ACL_MEM_MALLOC_HUGE_FIRST = 0
    MEMCPY_HOST_TO_DEVICE = 1
    MEMCPY_DEVICE_TO_HOST = 2
except ImportError:
    print("错误：未找到ACL模块，请配置环境变量")
    sys.exit(1)

# ===================== 核心配置（需与训练/转换对齐）=====================
CONFIG = {
    'MODEL_PATH': './emotion_cnn.om',          # 转换后的OM模型路径
    'DATA_PATH': './fer2013.csv',  # 测试数据路径
    'DEVICE_ID': 0,                            # NPU设备ID
    'NUM_CLASSES': 7,                          # 情绪类别数（0-6）
    'PROGRESS_STEP': 100,                      # 进度显示步长
    # 预处理配置（必须与训练时完全一致！）
    'PREPROCESS': {
        'mean': 0.0,       # 训练时的均值（如无则0）
        'std': 1.0,        # 训练时的标准差（如无则1）
        'scale': 1.0/255.0 # 像素归一化系数（训练时用/255则此处为1/255）
    },
    'EXPECT_INPUT_SIZE': 48*48*1*4  # 预期输入大小：48x48x1(float32)=9216字节
}

# 全局资源管理（避免重复初始化）
g_resources = {
    'context': None,
    'model_id': None,
    'model_desc': None,
    'input_buf': None,
    'output_buf': None,
    'input_size': 0,
    'output_size': 0
}

# 情绪标签映射（必须和训练代码完全一致！）
EMOTION_LABELS = {
    0: "生气", 1: "厌恶", 2: "恐惧",
    3: "开心", 4: "中性", 5: "悲伤", 6: "惊讶"
}


def stable_softmax(x):
    """数值稳定的Softmax（避免溢出）"""
    x_max = np.max(x)
    exp_x = np.exp(x - x_max)
    sum_exp_x = np.sum(exp_x)
    return exp_x / sum_exp_x if sum_exp_x != 0 else np.ones_like(x)/len(x)


def align_train_preprocess(pixel_array):
    """与训练完全对齐的预处理逻辑（核心！）"""
    img = pixel_array.astype(np.float32) * CONFIG['PREPROCESS']['scale']
    img = (img - CONFIG['PREPROCESS']['mean']) / CONFIG['PREPROCESS']['std']
    img = np.expand_dims(img, axis=0)  # batch维度
    img = np.expand_dims(img, axis=0)  # 通道维度

    # 调试信息（随机打印）
    if np.random.rand() < 0.001:
        print(f"\n【预处理调试】均值：{np.mean(img):.4f}，最大值：{np.max(img):.4f}，最小值：{np.min(img):.4f}")
        print(f"【预处理调试】形状：{img.shape}，字节数：{img.nbytes}")

    return img


def safe_acl_init():
    """安全初始化ACL和OM模型 + 输入大小验证"""
    try:
        if not os.path.exists(CONFIG['MODEL_PATH']):
            raise FileNotFoundError(f"OM模型不存在：{CONFIG['MODEL_PATH']}")

        # 初始化ACL
        ret = acl.init()
        ret = ret[-1] if isinstance(ret, (tuple, list)) else ret
        if ret != 0:
            raise RuntimeError(f"ACL初始化失败，错误码：{ret}")

        # 设置NPU设备
        ret = acl.rt.set_device(CONFIG['DEVICE_ID'])
        ret = ret[-1] if isinstance(ret, (tuple, list)) else ret
        if ret != 0:
            raise RuntimeError(f"设置NPU设备失败，错误码：{ret}")

        # 创建设备上下文
        create_ret = acl.rt.create_context(CONFIG['DEVICE_ID'])
        if isinstance(create_ret, (tuple, list)):
            g_resources['context'], ret = create_ret
        else:
            g_resources['context'] = create_ret
            ret = 0
        if ret != 0 or not g_resources['context']:
            raise RuntimeError(f"创建上下文失败，错误码：{ret}")

        # 加载OM模型
        load_ret = acl.mdl.load_from_file(CONFIG['MODEL_PATH'])
        if isinstance(load_ret, (tuple, list)):
            g_resources['model_id'], ret = load_ret
        else:
            g_resources['model_id'] = load_ret
            ret = 0
        if ret != 0 or not g_resources['model_id']:
            raise RuntimeError(f"加载OM模型失败，错误码：{ret}")

        # 获取模型描述和输入输出大小
        g_resources['model_desc'] = acl.mdl.create_desc()
        acl.mdl.get_desc(g_resources['model_desc'], g_resources['model_id'])
        g_resources['input_size'] = acl.mdl.get_input_size_by_index(g_resources['model_desc'], 0)
        g_resources['output_size'] = acl.mdl.get_output_size_by_index(g_resources['model_desc'], 0)

        # 输入大小验证
        print(f"\n⚠️  输入大小验证：")
        print(f"  模型要求输入大小：{g_resources['input_size']} 字节")
        print(f"  预期输入大小（48x48x1xfloat32）：{CONFIG['EXPECT_INPUT_SIZE']} 字节")
        if g_resources['input_size'] != CONFIG['EXPECT_INPUT_SIZE']:
            print(f"❌ 输入大小不匹配！可能是ONNX导出/ATC转换时输入形状错误")
            g_resources['input_size'] = CONFIG['EXPECT_INPUT_SIZE']

        # 申请NPU设备内存
        malloc_ret = acl.rt.malloc(int(g_resources['input_size']), int(ACL_MEM_MALLOC_HUGE_FIRST))
        if isinstance(malloc_ret, (tuple, list)):
            g_resources['input_buf'], ret = malloc_ret
        else:
            g_resources['input_buf'] = malloc_ret
            ret = 0
        if ret != 0 or not g_resources['input_buf']:
            raise RuntimeError(f"申请输入内存失败，错误码：{ret}")

        malloc_ret = acl.rt.malloc(int(g_resources['output_size']), int(ACL_MEM_MALLOC_HUGE_FIRST))
        if isinstance(malloc_ret, (tuple, list)):
            g_resources['output_buf'], ret = malloc_ret
        else:
            g_resources['output_buf'] = malloc_ret
            ret = 0
        if ret != 0 or not g_resources['output_buf']:
            raise RuntimeError(f"申请输出内存失败，错误码：{ret}")

        print(f"✅ ACL初始化成功")
        print(f"  - 模型输入大小：{g_resources['input_size']} 字节")
        print(f"  - 模型输出大小：{g_resources['output_size']} 字节")
        return True

    except Exception as e:
        print(f"❌ ACL初始化失败：{e}")
        traceback.print_exc()
        safe_acl_cleanup()
        return False


def om_model_infer(img_array):
    """OM模型单样本推理"""
    try:
        img_bytes = img_array.tobytes()
        actual_size = len(img_bytes)
        if actual_size > g_resources['input_size']:
            img_bytes = img_bytes[:g_resources['input_size']]
            print(f"⚠️  输入数据过长，已截断至{g_resources['input_size']}字节")
        elif actual_size < g_resources['input_size']:
            print(f"⚠️  输入数据过短（{actual_size}字节 < {g_resources['input_size']}字节），补零可能导致失真")
            img_bytes = img_bytes.ljust(g_resources['input_size'], b'\x00')

        img_buffer = ctypes.create_string_buffer(img_bytes)
        src_ptr = ctypes.addressof(img_buffer)

        # 主机→NPU内存拷贝
        memcpy_ret = acl.rt.memcpy(
            int(g_resources['input_buf']),
            int(g_resources['input_size']),
            int(src_ptr),
            int(len(img_bytes)),
            int(MEMCPY_HOST_TO_DEVICE)
        )
        ret = memcpy_ret[-1] if isinstance(memcpy_ret, (tuple, list)) else memcpy_ret
        if ret != 0:
            raise RuntimeError(f"内存拷贝失败，错误码：{ret}")

        # 创建数据集和缓冲区
        dataset_in = acl.mdl.create_dataset()
        dataset_out = acl.mdl.create_dataset()

        # 输入缓冲区
        buf_ret = acl.create_data_buffer(int(g_resources['input_buf']), int(len(img_bytes)))
        input_buf_obj = buf_ret[0] if isinstance(buf_ret, (tuple, list)) else buf_ret
        if not input_buf_obj:
            raise RuntimeError("创建输入缓冲区失败")

        add_ret = acl.mdl.add_dataset_buffer(dataset_in, input_buf_obj)
        ret = add_ret[-1] if isinstance(add_ret, (tuple, list)) else add_ret
        if ret != 0:
            raise RuntimeError(f"添加输入缓冲区失败，错误码：{ret}")

        # 输出缓冲区
        buf_ret = acl.create_data_buffer(int(g_resources['output_buf']), int(g_resources['output_size']))
        output_buf_obj = buf_ret[0] if isinstance(buf_ret, (tuple, list)) else buf_ret
        if not output_buf_obj:
            raise RuntimeError("创建输出缓冲区失败")

        add_ret = acl.mdl.add_dataset_buffer(dataset_out, output_buf_obj)
        ret = add_ret[-1] if isinstance(add_ret, (tuple, list)) else add_ret
        if ret != 0:
            raise RuntimeError(f"添加输出缓冲区失败，错误码：{ret}")

        # 执行NPU推理
        exec_ret = acl.mdl.execute(g_resources['model_id'], dataset_in, dataset_out)
        ret = exec_ret[-1] if isinstance(exec_ret, (tuple, list)) else exec_ret
        if ret != 0:
            raise RuntimeError(f"推理执行失败，错误码：{ret}")

        # NPU→主机内存拷贝
        out_buffer = ctypes.create_string_buffer(g_resources['output_size'])
        dst_ptr = ctypes.addressof(out_buffer)
        memcpy_ret = acl.rt.memcpy(
            int(dst_ptr),
            int(g_resources['output_size']),
            int(g_resources['output_buf']),
            int(g_resources['output_size']),
            int(MEMCPY_DEVICE_TO_HOST)
        )
        ret = memcpy_ret[-1] if isinstance(memcpy_ret, (tuple, list)) else memcpy_ret
        if ret != 0:
            raise RuntimeError(f"输出拷贝失败，错误码：{ret}")

        # 解析推理结果
        output_data = np.frombuffer(out_buffer.raw, dtype=np.float32)
        output_data = output_data[:CONFIG['NUM_CLASSES']]
        pred_probs = stable_softmax(output_data)
        pred_label = np.argmax(pred_probs).item()

        # 调试信息（随机打印）
        if np.random.rand() < 0.01:
            print(f"\n【推理调试】输出logits：{output_data}")
            print(f"【推理调试】输出概率：{pred_probs}")
            print(f"【推理调试】预测标签：{pred_label}")

        # 释放临时资源
        acl.destroy_data_buffer(input_buf_obj)
        acl.destroy_data_buffer(output_buf_obj)
        acl.mdl.destroy_dataset(dataset_in)
        acl.mdl.destroy_dataset(dataset_out)

        return pred_label

    except Exception as e:
        # 异常处理
        try:
            if 'dataset_in' in locals():
                acl.mdl.destroy_dataset(dataset_in)
            if 'dataset_out' in locals():
                acl.mdl.destroy_dataset(dataset_out)
            if 'input_buf_obj' in locals():
                acl.destroy_data_buffer(input_buf_obj)
            if 'output_buf_obj' in locals():
                acl.destroy_data_buffer(output_buf_obj)
        except:
            pass
        print(f"❌ 单样本推理失败：{e}")
        return -1


def load_samples_by_type(data_type):
    """加载指定类型的样本"""
    try:
        if not os.path.exists(CONFIG['DATA_PATH']):
            raise FileNotFoundError(f"数据文件不存在：{CONFIG['DATA_PATH']}")

        samples = []
        label_dist = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0}

        import csv
        with open(CONFIG['DATA_PATH'], "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader)
            emo_idx = header.index("emotion")
            pix_idx = header.index("pixels")
            use_idx = header.index("Usage") if "Usage" in header else -1

            for row in reader:
                if len(row) <= max(emo_idx, pix_idx):
                    continue

                # 数据类型筛选
                if use_idx >= 0:
                    usage = row[use_idx]
                    if data_type == 'train' and usage != "Training":
                        continue
                    if data_type == 'test' and usage not in ["PrivateTest", "PublicTest"]:
                        continue

                # 解析数据
                pixel_list = list(map(int, row[pix_idx].split()))
                pixel_array = np.array(pixel_list).reshape(48, 48)
                true_label = int(row[emo_idx])
                img_tensor = align_train_preprocess(pixel_array)
                samples.append((img_tensor, true_label, pixel_array))
                label_dist[true_label] += 1

        print(f"\n📊 {data_type}样本加载完成")
        print(f"  - 总样本数：{len(samples)}")
        print(f"  - 标签分布：")
        for label, cnt in label_dist.items():
            print(f"    {EMOTION_LABELS[label]}({label})：{cnt} 条")

        # 简单验证
        if samples:
            test_idx = 0
            img_tensor, true_label, _ = samples[test_idx]
            om_pred = om_model_infer(img_tensor)
            print(f"\n🔍 OM验证（{data_type}样本{test_idx}）：")
            print(f"  真实标签：{true_label}({EMOTION_LABELS[true_label]})")
            print(f"  OM预测：{om_pred}({EMOTION_LABELS.get(om_pred, '未知')})")

        return samples

    except Exception as e:
        print(f"❌ 加载{data_type}样本失败：{e}")
        traceback.print_exc()
        return []


def evaluate_dataset(data_type):
    """评估指定类型数据集，返回评估结果字典"""
    samples = load_samples_by_type(data_type)
    if not samples:
        print(f"⚠️  没有可用的{data_type}样本，跳过评估")
        return None

    total = len(samples)
    correct = 0
    failed = 0
    label_stats = {i: {'total':0, 'correct':0} for i in range(7)}

    print(f"\n🚀 开始{data_type}数据集推理（共{total}条样本）")
    start_time = time.time()

    for idx, (img_tensor, true_label, _) in enumerate(samples):
        pred_label = om_model_infer(img_tensor)
        label_stats[true_label]['total'] += 1
        if pred_label == -1:
            failed += 1
        else:
            if pred_label == true_label:
                correct += 1
                label_stats[true_label]['correct'] += 1

        # 进度显示
        if (idx + 1) % CONFIG['PROGRESS_STEP'] == 0:
            elapsed = time.time() - start_time
            speed = (idx + 1) / elapsed if elapsed > 0 else 0
            acc = 100.0 * correct / (idx + 1 - failed) if (idx + 1 - failed) > 0 else 0
            print(f"  进度：[{idx+1}/{total}] | 失败：{failed} | 准确率：{acc:.2f}% | 速度：{speed:.2f}样本/秒")

    # 计算结果
    elapsed_total = time.time() - start_time
    valid = total - failed
    overall_acc = 100.0 * correct / valid if valid > 0 else 0

    # 打印该数据集的详细结果（保留中间信息）
    print(f"\n🎯 {data_type}数据集推理结果汇总")
    print(f"  - 总耗时：{elapsed_total:.2f} 秒")
    print(f"  - 平均速度：{total/elapsed_total:.2f} 样本/秒")
    print(f"  - 总样本：{total} | 失败：{failed} | 有效：{valid}")
    print(f"  - 总体准确率：{overall_acc:.2f}%")
    print(f"  按标签准确率：")
    for label in range(7):
        total_l = label_stats[label]['total']
        correct_l = label_stats[label]['correct']
        acc_l = 100.0 * correct_l / total_l if total_l > 0 else 0
        print(f"    {EMOTION_LABELS[label]}({label})：{correct_l}/{total_l} | {acc_l:.2f}%")
    print("--------------------------------------------------")

    # 返回关键指标用于最终汇总
    return {
        'type': data_type,
        'total': total,
        'failed': failed,
        'valid': valid,
        'accuracy': overall_acc
    }


def safe_acl_cleanup():
    """安全释放ACL和NPU资源"""
    try:
        if g_resources['output_buf']:
            acl.rt.free(int(g_resources['output_buf']))
            g_resources['output_buf'] = None
        if g_resources['input_buf']:
            acl.rt.free(int(g_resources['input_buf']))
            g_resources['input_buf'] = None

        if g_resources['model_desc']:
            acl.mdl.destroy_desc(g_resources['model_desc'])
            g_resources['model_desc'] = None

        if g_resources['model_id']:
            unload_ret = acl.mdl.unload(int(g_resources['model_id']))
            ret = unload_ret[-1] if isinstance(unload_ret, (tuple, list)) else unload_ret
            if ret != 0:
                print(f"⚠️  卸载模型警告，错误码：{ret}")
            g_resources['model_id'] = None

        if g_resources['context']:
            acl.rt.destroy_context(g_resources['context'])
            g_resources['context'] = None

        acl.rt.reset_device(CONFIG['DEVICE_ID'])
        acl.finalize()
        print(f"\n✅ ACL资源释放完成")

    except Exception as e:
        print(f"⚠️  资源释放警告：{e}")


if __name__ == '__main__':
    """主函数：初始化→评估→汇总打印"""
    print("=====================================")
    print("  OM模型推理验证（昇腾Atlas 200 DK）  ")
    print("=====================================")

    # 1. 初始化ACL和模型
    if not safe_acl_init():
        sys.exit(1)

    # 2. 评估各数据集并保存结果
    eval_results = []
    eval_results.append(evaluate_dataset('train'))    # 训练集评估
    eval_results.append(evaluate_dataset('test'))     # 测试集评估
    eval_results.append(evaluate_dataset('all'))      # 所有数据评估

    # 3. 释放资源
    safe_acl_cleanup()

    # 4. 最终汇总打印三种准确率
    print("\n=====================================")
    print("          准确率汇总对比           ")
    print("=====================================")
    for result in eval_results:
        if result:  # 跳过空结果（无样本的情况）
            print(f"{result['type']}集：")
            print(f"  总样本数：{result['total']} | 失败：{result['failed']} | 有效样本：{result['valid']}")
            print(f"  准确率：{result['accuracy']:.2f}%")
            print("-------------------------------------")

    print("\n🎉 所有数据集推理验证完成！")