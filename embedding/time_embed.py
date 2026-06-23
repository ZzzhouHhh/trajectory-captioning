"""
时间节点嵌入生成。

生成 (weekday, hour) -> embedding 的映射，
用于初始化时间-子类别GCN的时间节点特征。

编码方式：独热编码(7+24) + 周期性编码(4) = 35维
"""

import numpy as np
import pickle
from tqdm import tqdm


def time_encoding_full(week, hour):
    """
    完整时间编码：独热 + 正弦余弦周期编码。

     公式(1)：将小时转为单位圆上的弧度进行sin/cos编码。
    """
    week_one_hot = np.eye(7)[week]
    hour_one_hot = np.eye(24)[hour]

    sin_week = np.sin(2 * np.pi * week / 7)
    cos_week = np.cos(2 * np.pi * week / 7)
    sin_hour = np.sin(2 * np.pi * hour / 24)
    cos_hour = np.cos(2 * np.pi * hour / 24)

    return np.concatenate([week_one_hot, hour_one_hot,
                           [sin_week, cos_week, sin_hour, cos_hour]])


def generate_time_embeddings():
    """生成所有168个时间节点(7天×24小时)的嵌入，保存为pkl。"""
    time_embeddings = {}
    for weekday in range(7):
        for hour in range(24):
            time_embeddings[(weekday, hour)] = time_encoding_full(weekday, hour)
    return time_embeddings


def get_time_embedding(weekday, hour, time_embeddings):
    """获取指定时间的嵌入向量"""
    key = (weekday, hour)
    if key in time_embeddings:
        return time_embeddings[key]
    else:
        return np.zeros(35)


def save_time_embeddings(time_embeddings, save_path):
    with open(save_path, 'wb') as f:
        pickle.dump(time_embeddings, f)
    print(f"时间嵌入已保存至 {save_path}")


def load_time_embeddings(path):
    with open(path, 'rb') as f:
        return pickle.load(f)
