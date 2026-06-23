import pandas as pd
import numpy as np
from tqdm import tqdm

from utils.spatial import haversine_np
from utils.temporal import weekday_map


# ========== POI 最近邻匹配（遍历版） ==========

def match_nearest_poi(trajectory_df, poi_df):
    """
    对每条轨迹的上下车点匹配最近的POI。
    双重循环遍历实现 —— 适合中小规模数据。
    """
    # 创建新的列用于存储匹配的 POI ID
    trajectory_df['matched_pickup_poi_id'] = None
    trajectory_df['matched_dropoff_poi_id'] = None

    for i in range(len(trajectory_df)):
        pickup_lon = trajectory_df.loc[i, 'pickup_longitude']
        pickup_lat = trajectory_df.loc[i, 'pickup_latitude']
        dropoff_lon = trajectory_df.loc[i, 'dropoff_longitude']
        dropoff_lat = trajectory_df.loc[i, 'dropoff_latitude']

        min_pickup_distance = float('inf')
        min_dropoff_distance = float('inf')
        matched_pickup_poi_id = None
        matched_dropoff_poi_id = None

        for j in range(len(poi_df)):
            poi_lon = poi_df.loc[j, 'longitude']
            poi_lat = poi_df.loc[j, 'latitude']
            poi_id = poi_df.loc[j, 'poi_id']

            pickup_distance = haversine_np(pickup_lon, pickup_lat, poi_lon, poi_lat)
            if pickup_distance < min_pickup_distance:
                min_pickup_distance = pickup_distance
                matched_pickup_poi_id = poi_id

            dropoff_distance = haversine_np(dropoff_lon, dropoff_lat, poi_lon, poi_lat)
            if dropoff_distance < min_dropoff_distance:
                min_dropoff_distance = dropoff_distance
                matched_dropoff_poi_id = poi_id

        trajectory_df.at[i, 'matched_pickup_poi_id'] = matched_pickup_poi_id
        trajectory_df.at[i, 'matched_dropoff_poi_id'] = matched_dropoff_poi_id

    return trajectory_df


# ========== POI匹配（向量化版本——更快但占内存） ==========

def match_nearest_poi_vectorized(trajectory_df, poi_df):
    poi_coords = poi_df[['latitude', 'longitude']].values

    def find_closest_poi(lat, lon):
        distances = np.array([
            haversine_np(lon, lat, p_lon, p_lat)
            for p_lat, p_lon in poi_coords
        ])
        return poi_df.iloc[distances.argmin()]['poi_id']

    for idx, row in tqdm(trajectory_df.iterrows(), total=len(trajectory_df),
                         desc="匹配POIs"):
        trajectory_df.at[idx, 'matched_pickup_poi_id'] = find_closest_poi(
            row['pickup_latitude'], row['pickup_longitude']
        )
        trajectory_df.at[idx, 'matched_dropoff_poi_id'] = find_closest_poi(
            row['dropoff_latitude'], row['dropoff_longitude']
        )

    return trajectory_df


# ========== POI信息与轨迹拼接 ==========

def join_poi_info_to_trajectory(csv1, csv2):
    """
    将POI的详细信息（name, category, venueCategory）拼接到轨迹表中。
    csv1: 轨迹表（含 pickup_poi_id, dropoff_poi_id）
    csv2: POI信息表
    """
    # 添加星期映射
    csv1['pickup_week'] = csv1['pickup_weekday'].map(weekday_map)

    # 重命名csv2以便匹配pickup信息
    csv2_pickup = csv2.rename(columns={
        "poi_id": "pickup_poi_id",
        "venueCategory": "pickup_venueCategory",
        "name": "pickup_name",
        "category": "pickup_category"
    })
    csv1 = csv1.merge(csv2_pickup, on="pickup_poi_id", how="left")

    # 重命名csv2以便匹配dropoff信息
    csv2_dropoff = csv2.rename(columns={
        "poi_id": "dropoff_poi_id",
        "venueCategory": "dropoff_venueCategory",
        "name": "dropoff_name",
        "category": "dropoff_category"
    })
    csv1 = csv1.merge(csv2_dropoff, on="dropoff_poi_id", how="left")

    # 添加trajectory_id
    csv1['trajectory_id'] = range(1, len(csv1) + 1)

    output_columns = [
        'trajectory_id', 'pickup_venueCategory', 'pickup_name', 'pickup_category',
        'dropoff_venueCategory', 'dropoff_name', 'dropoff_category',
        'passenger_count', 'pickup_hour', 'dropoff_hour', 'pickup_week'
    ]
    csv3 = csv1[output_columns]
    return csv3


# ========== 合并 co 特征列 ==========

def merge_co_features(traj_csv, co_csv):
    """
    将上下文增强计算出的 co_pickup_0~co_pickup_8, co_dropoff_0~co_dropoff_8
    共18列合并到轨迹表中。
    """
    if len(traj_csv) != len(co_csv):
        raise ValueError("两个CSV文件的行数不一致，无法一一对应合并！")

    extra_columns = [
        'co_pickup_0', 'co_dropoff_0', 'co_pickup_1', 'co_dropoff_1',
        'co_pickup_2', 'co_dropoff_2', 'co_pickup_3', 'co_dropoff_3',
        'co_pickup_4', 'co_dropoff_4', 'co_pickup_5', 'co_dropoff_5',
        'co_pickup_6', 'co_dropoff_6', 'co_pickup_7', 'co_dropoff_7',
        'co_pickup_8', 'co_dropoff_8'
    ]
    existed_cols = [c for c in extra_columns if c in co_csv.columns]
    merged = pd.concat([traj_csv, co_csv[existed_cols]], axis=1)
    print("合并完成，co特征已添加到轨迹表中。")
    return merged
