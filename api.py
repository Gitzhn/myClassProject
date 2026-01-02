"""
简化版 API 启动脚本：复用同目录的 emotion_api 推理与路由。
运行本文件即可启动 OM 模型的 Flask 推理服务。
"""
import sys
from pathlib import Path

# 当前文件所在目录（即 model 目录）
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

try:
    from emotion_api import app, CONFIG, global_acl_init, global_acl_cleanup  # type: ignore
except Exception as exc:  # pragma: no cover
    sys.stderr.write(f"导入 emotion_api 失败：{exc}\n")
    sys.exit(1)


@app.after_request
def add_cors_headers(resp):  # type: ignore
    resp.headers["Access-Control-Allow-Origin"] = "*"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
    resp.headers["Access-Control-Allow-Methods"] = "POST, OPTIONS"
    return resp


def update_config_paths() -> None:
    """确保模型路径为绝对路径，避免相对路径在不同 cwd 下失效。"""
    model_path = (BASE_DIR / "emotion_cnn.om").resolve()
    CONFIG["MODEL_PATH"] = str(model_path)


def main(port: int = 5000) -> None:
    update_config_paths()
    if not global_acl_init():
        sys.exit(1)
    try:
        app.run(host="0.0.0.0", port=port, threaded=False)
    except KeyboardInterrupt:
        sys.stderr.write("\n收到中断信号，准备退出...\n")
    finally:
        global_acl_cleanup()


if __name__ == "__main__":
    p = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
    main(p)