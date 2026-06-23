import numpy as np
import pandas as pd
from math import radians, sin, cos, sqrt, atan2

# 地球半径（单位：千米）
EARTH_RADIUS = 6371


# Haversine 公式计算两点之间的距离（单位：米）
def haversine_distance(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    distance = EARTH_RADIUS * c
    return distance * 1000  # 返回米


# Haversine 的 numpy 版本 —— 另一个实现
def haversine_np(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 6371000  # 地球半径（单位：米）
    return c * r


# 添加随机偏移（模拟定位误差）
def add_random_offset(lat, lon, offset_meters=200):
    lat_deg_to_m = 111319.9  # 每纬度一度约为 111319.9 米
    lon_deg_to_m = 111319.9 * cos(radians(lat))  # 经度随纬度变化
    lat_offset = np.random.uniform(-offset_meters, offset_meters)
    lon_offset = np.random.uniform(-offset_meters, offset_meters)
    new_lat = lat + (lat_offset / lat_deg_to_m)
    new_lon = lon + (lon_offset / lon_deg_to_m)
    return new_lat, new_lon


# 给经纬度加上随机扰动（训练时数据增强用）
def add_random_noise_to_location(df, noise_radius=150):
    """
    给经纬度加上 noise_radius 米的随机扰动
    """
    noise_lat = np.random.normal(0, noise_radius / 111000, size=df.shape[0])
    noise_lon = np.random.normal(0, noise_radius / (111000 * np.cos(np.radians(df['lat']))), size=df.shape[0])
    df['lat'] += noise_lat
    df['lon'] += noise_lon
    return df


# 网格索引相关函数
def create_grid_index(df, grid_size=0.002):
    df['grid_id'] = df.apply(lambda row: f"{int(row['lat'] // grid_size)}_{int(row['lon'] // grid_size)}", axis=1)
    return df


def find_grid(lat, lon, grid_size=0.002):
    return f"{int(lat // grid_size)}_{int(lon // grid_size)}"


def get_neighboring_grids(grid_id, grid_size=0.002):
    lat_idx, lon_idx = map(int, grid_id.split('_'))
    neighboring_grids = []
    for i in range(-1, 2):
        for j in range(-1, 2):
            new_lat_idx = lat_idx + i
            new_lon_idx = lon_idx + j
            neighboring_grids.append(f"{new_lat_idx}_{new_lon_idx}")
    return neighboring_grids
