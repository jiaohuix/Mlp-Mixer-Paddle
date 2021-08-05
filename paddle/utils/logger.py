# -*- coding:utf-8 -*-
import os
import logging

def get_logger(loggername,save_path='.'):
    # 创建一个logger
    logger = logging.getLogger(loggername)
    logger.setLevel(logging.DEBUG)
    save_path = save_path

    # 创建一个handler，用于写入日志文件
    log_path = save_path + "/logs/"  # 指定文件输出路径，注意logs是个文件夹，一定要加上/，不然会导致输出路径错误，把logs变成文件名的一部分了
    if not os.path.exists(log_path):
        os.mkdir(log_path)
    logname = log_path + '{}.log'.format(loggername)  # 指定输出的日志文件名
    fh = logging.FileHandler(logname, encoding='utf-8')  # 指定utf-8格式编码，避免输出的日志文本乱码
    fh.setLevel(logging.DEBUG)

    # 创建一个handler，用于将日志输出到控制台
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)

    # 定义handler的输出格式
    formatter = logging.Formatter('%(asctime)s | %(name)s: %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    # 给logger添加handler
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger
logger=get_logger('Cifar10')

