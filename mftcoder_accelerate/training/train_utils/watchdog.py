"""
 @author Chaoyu Chen
 watchdog which supports callback functions for checkpointing
"""
import time
import os
import argparse
import json
import logging
import sys
import pprint

# 配置logger
# 设置日志级别为DEBUG，这意味着DEBUG及以上级别的所有日志都会被捕获
# 默认的日志格式包含了时间戳、日志级别、消息和日志发生地点的相关信息
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")

# 获取一个logger实例
logger = logging.getLogger(__name__)


def func(ckpt_path, ckpt_name, base_model_path, model_type):
    pass


def ckpt_callback(callback, ckpt_path, ckpt_name, base_model_path, model_type):
    lora = os.path.isfile(os.path.join(ckpt_path, "adapter_config.json"))
    logger.info(f"use lora inference: {lora}")
    logger.info(f"submit: {ckpt_name}")
    callback(ckpt_path, ckpt_name, base_model_path, model_type)


def main(ckpt_dir: str, base_model_path: str, model_type):
    # 单ckpt评测
    if ckpt_dir.split("/")[-1].startswith(("epoch_", "step_", "checkpoint-")):
        ckpt_path = ckpt_dir
        ckpt_name = ckpt_dir.split("/")[-2] + "-" + ckpt_dir.split("/")[-1]
        ckpt_callback(func, ckpt_path, ckpt_name, base_model_path, model_type)
        return

    # 进入监控循环
    stop_limit = 100
    stop_num = 0
    # 读取history
    history_path = os.path.join(ckpt_dir, "watchdog")
    if os.path.exists(history_path):
        with open(history_path, "r") as f:
            history = json.load(f)
            known_dirs = set(history["submitted"])
            logger.info(f"submitted: {known_dirs}")
    else:
        logger.info(f"history: {history_path} 不存在")
        # 初始化已知目录状态
        # known_dirs = {name for name in os.listdir(ckpt_dir) if os.path.isdir(os.path.join(ckpt_dir, name)) and name.startswith(('epoch_', 'step_', "checkpoint-"))}
        known_dirs = set()
        logger.info(f"initialized: {known_dirs}")

    # 主监控循环
    try:
        while True:
            # 兼容ckpt_dir没有被创建
            if os.path.exists(ckpt_dir):
                current_dirs = {
                    name
                    for name in os.listdir(ckpt_dir)
                    if
                    os.path.isdir(os.path.join(ckpt_dir, name)) and name.startswith(("epoch_", "step_", "checkpoint-"))
                }
                # ckpt_dir存在，但还没有任何submits的时候，打印一次训练的args
                if not os.path.exists(history_path) and os.path.exists(os.path.join(ckpt_dir, "args.json")):
                    with open(os.path.join(ckpt_dir, "args.json"), "r") as f:
                        pprint.pprint(json.load(f))
                    sys.stdout.flush()
            else:
                logger.info("ckpt dir not created, training may not strart")
                time.sleep(300)  # 定时间隔 5min
                continue
            # 检测新添加的文件夹
            new_dirs = current_dirs - known_dirs
            logger.info(f"new_dirs: {new_dirs}")
            if not new_dirs:
                stop_num += 1
            else:
                stop_num = 0
                time.sleep(60)
            if stop_num > stop_limit:
                break
            for new_dir in new_dirs:
                # 检查是否以 'epoch_' 或 'step_' 开头
                if new_dir.startswith(("epoch_", "step_", "checkpoint-")):
                    ckpt_path = os.path.join(ckpt_dir, new_dir)
                    ckpt_name = ckpt_dir.split("/")[-1] + "-" + new_dir
                    ckpt_callback(func, ckpt_path, ckpt_name, base_model_path, model_type)

            # 更新已知目录列表
            known_dirs = current_dirs
            history = {"submitted": list(known_dirs)}
            with open(history_path, "w") as f:
                json.dump(history, f, indent=2)
            time.sleep(300)  # 定时间隔 5min
    except KeyboardInterrupt:
        print("监控停止")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="watchdog参数.")
    parser.add_argument(
        "--base_model_path",
        type=str,
    )
    parser.add_argument("--ckpt_dir", type=str)

    parser.add_argument("--model_type", type=str)

    args = parser.parse_args()
    main(
        ckpt_dir=args.ckpt_dir,
        base_model_path=args.base_model_path,
        model_type=args.model_type,
    )
