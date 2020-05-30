"""
生成序列，为输入准备数据
"""
import pandas as pd
import numpy as np
import math


def get_item(path_session):
    """
    :return:
    tar会话最后一个用于求损失的
    item会话序列
    user用户
    """
    items = []
    users = []
    tar = []
    with open(path_session, encoding='utf-8') as f:
        for line in f:
            session = [int(s) - 1 for s in line.strip().split(',')]
            if len(session[1:-1]) == 0:
                continue
            users.append(session[0] + 1)
            items.append(session[1:-1])
            tar.append(session[-1])
    return np.array(items), np.array(tar), np.array(users)


def generate_batch(file_path, batch_size, shuffle=False):
    items, tar, user = get_item(file_path)
    length = len(tar)
    if shuffle:
        shuffled = np.arange(length)
        np.random.shuffle(shuffled)
        items = items[shuffled]
        tar = tar[shuffled]
        user = user[shuffled]
    n_batch = int(math.ceil(length / batch_size))
    slice = np.split(np.arange(n_batch * batch_size), n_batch)
    slice[-1] = np.arange(length - batch_size, length)
    for index in slice:
        # 计数每个会话的长度
        n_session = [len(item) for item in items[index]]
        # 返回这一批次的会话
        max_size = np.max(n_session)
        min_size = np.min(n_session)
        n_item = [item + (max_size - len(item)) * [0] for item in items[index]]
        yield n_item, tar[index], user[index], n_session


def get_context(file_path):
    """
    加载github代码库的内容信息
    :param file_path:
    :return:
    """
    df = pd.read_csv(file_path, header=None, index_col=False)
    df = df.sort_values(by=0, axis=0)
    print(df.shape[0])
    bb = df.iloc[:, 1:].values
    return bb


def normalization(x):
    """"
    归一化到区间{0,1]
    返回副本
    """
    shape = x.shape
    _range = np.max(x, axis=1) - np.min(x, axis=1)
    ss = (x - np.min(x, axis=1).reshape([shape[0], 1])) / _range.reshape([shape[0], 1])
    return ss


def standardization(x):
    """
    z-score normalization
    :param x:
    :return:
    """
    shape = x.shape
    mean = np.mean(x, axis=1).reshape([shape[0], 1])
    std = np.std(x, axis=1).reshape([shape[0], 1])
    x_hat = x - mean
    return x_hat / std


def means(x):
    shape = x.shape
    total = np.sum(x, axis=1)
    total = total.reshape([shape[0], 1])
    return x / total
