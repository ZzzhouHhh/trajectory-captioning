import numpy as np
import pandas as pd

# 星期映射
weekday_map = {0: "Monday", 1: "Tuesday", 2: "Wednesday", 3: "Thursday",
               4: "Friday", 5: "Saturday", 6: "Sunday"}


# 时间编码函数：独热编码 + 周期性（正弦余弦）编码
def time_encoding(week, hour):
    """
    将时间（星期几、小时）编码为独热编码 + 周期性（正弦余弦）编码。
    将小时转换为单位圆上的弧度，用 sin/cos 保持时间连续性。
    :param week: 星期几 (0-6)
    :param hour: 小时 (0-23)
    :return: 时间编码向量，维度 = 7 + 24 + 4 = 35
    """
    # 周时间的独热编码（长度为7）
    week_one_hot = np.eye(7)[week]

    # 小时时间的独热编码（长度为24）
    hour_one_hot = np.eye(24)[hour]

    # 周时间的周期性编码（正弦余弦）
    sin_week = np.sin(2 * np.pi * week / 7)
    cos_week = np.cos(2 * np.pi * week / 7)

    # 小时时间的周期性编码（正弦余弦）—— 
    sin_hour = np.sin(2 * np.pi * hour / 24)
    cos_hour = np.cos(2 * np.pi * hour / 24)

    # 组合独热编码和周期性编码
    time_embedding = np.concatenate([week_one_hot, hour_one_hot, [sin_week, cos_week, sin_hour, cos_hour]])

    return time_embedding


def convert_is_weekend(value):
    if isinstance(value, bool):
        return 1 if value else 0
    elif isinstance(value, str) and value.lower() == 'true':
        return 1
    elif isinstance(value, str) and value.lower() == 'false':
        return 0
    else:
        return 1 if value >= 5 else 0


# 增强的时间上下文表示
# T_t = [sin(hour), cos(hour), is_weekend, weekday]
def extract_enhanced_temporal(hour, weekday):
    sin_h = np.sin(2 * np.pi * hour / 24)
    cos_h = np.cos(2 * np.pi * hour / 24)
    is_weekend = 1 if weekday >= 5 else 0
    return [sin_h, cos_h, is_weekend, weekday]


def preprocess_temporal_data(trips_df):
    # 将datetime列转换为datetime类型，并提取小时信息
    trips_df['tpep_pickup_datetime'] = pd.to_datetime(trips_df['tpep_pickup_datetime'])
    trips_df['pickup_hour'] = trips_df['tpep_pickup_datetime'].dt.hour
    trips_df['tpep_dropoff_datetime'] = pd.to_datetime(trips_df['tpep_dropoff_datetime'])
    trips_df['drop_hour'] = trips_df['tpep_dropoff_datetime'].dt.hour

    # 判断是否是周末
    trips_df['is_weekend'] = trips_df['tpep_pickup_datetime'].dt.weekday >= 5

    return trips_df
