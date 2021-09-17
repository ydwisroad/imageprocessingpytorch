import logging
import os
import sys
import time


def get_logger(logdir):

    if not os.path.exists(logdir):
        os.makedirs(logdir)
    logname = f'run-{time.strftime("%Y-%m-%d-%H-%M")}.log'  # Python time strftime() 返回以可读字符串表示的当地时间，格式由参数format决定
    log_file = os.path.join(logdir, logname)

    # create log
    logger = logging.getLogger('train')     # log初始化
    logger.setLevel(logging.INFO)       # 设置log级别

    # Formatter 设置日志输出格式
    formatter = logging.Formatter('%(asctime)s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    # StreamHandler 日志输出1 -> 到控制台
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    # FileHandler 日志输出2 -> 保存到文件log_file
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger