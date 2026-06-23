import pandas as pd
import numpy as np
from tqdm import tqdm

from utils.spatial import create_grid_index, find_grid
from utils.temporal import preprocess_temporal_data
from .popularity import calculate_popularity
from .co_calculation import calculate_co, parallel_calculate_co


# ========== 主流程：轨迹上下文增强 ==========

def augment_trajectory_context(
    trips_df,
    checkin_df,
    num_categories=9,
    w=0.4,
    grid_size=0.002,
    use_parallel=True
):
    """
    对轨迹数据进行完整的时空上下文增强。

    输出新增列：
    - co_pickup_{k}, co_dropoff_{k}: 上下车点的时空上下文 (k=0..8)
    - pp_pickup_{k}, pp_dropoff_{k}: 时间流行度
    - u_pickup_{k}, u_dropoff_{k}: 类别独特性（各函数独立计算）

    参数:
        trips_df: 轨迹DataFrame
        checkin_df: 签到数据DataFrame（需含lat, lon, classification, poi_id列）
        num_categories: POI类别数量（9）
        w (omega): P_k和U_k的权重平衡参数，默认0.4
        grid_size: 空间网格大小（默认0.002度≈220m）
        use_parallel: 是否使用多进程并行计算
    """
    # 1. 预处理：提取时间特征
    trips_df = preprocess_temporal_data(trips_df)

    # 2. 创建网格索引
    checkin_df = create_grid_index(checkin_df)
    trips_df['pickup_grid_id'] = trips_df.apply(
        lambda row: find_grid(row['pickup_latitude'], row['pickup_longitude']), axis=1)
    trips_df['dropoff_grid_id'] = trips_df.apply(
        lambda row: find_grid(row['dropoff_latitude'], row['dropoff_longitude']), axis=1)

    # 3. 统计每个网格中的签到点数量
    checkin_counts = checkin_df.groupby(
        ['grid_id', 'classification', 'hour']).size().reset_index(name='count')

    # 4. 计算每个类别的co值
    for classification in range(num_categories):
        co_pickup_col = f'co_pickup_{classification}'
        co_dropoff_col = f'co_dropoff_{classification}'

        if use_parallel:
            trips_df[co_pickup_col] = parallel_calculate_co(
                trips_df, classification, checkin_df, checkin_counts, pickup=True, w=w
            )
            trips_df[co_dropoff_col] = parallel_calculate_co(
                trips_df, classification, checkin_df, checkin_counts, pickup=False, w=w
            )
        else:
            # 串行版本
            trips_df[co_pickup_col] = trips_df.apply(
                lambda row: calculate_co(row, classification, checkin_df, checkin_counts,
                                         pickup=True, w=w), axis=1)
            trips_df[co_dropoff_col] = trips_df.apply(
                lambda row: calculate_co(row, classification, checkin_df, checkin_counts,
                                         pickup=False, w=w), axis=1)

    # 5. 同时计算pp和u值（用于分析和消融实验）
    for classification in range(num_categories):
        trips_df[f'pp_pickup_{classification}'] = trips_df.apply(
            lambda row: calculate_popularity(row, classification, checkin_counts, pickup=True),
            axis=1)
        trips_df[f'pp_dropoff_{classification}'] = trips_df.apply(
            lambda row: calculate_popularity(row, classification, checkin_counts, pickup=False),
            axis=1)

    # 保存计算结果示例
    print(trips_df[['tpep_pickup_datetime', 'pickup_latitude', 'pickup_longitude',
                    'tpep_dropoff_datetime', 'dropoff_latitude', 'dropoff_longitude',
                    'is_weekend'] + [f'co_dropoff_{i}' for i in range(num_categories)]].head())

    return trips_df


# ========== 消融实验：不包含co的版本（w/o c）==========
# def augment_without_co(trips_df, checkin_df):
#     trips_df = preprocess_temporal_data(trips_df)
#     checkin_df = create_grid_index(checkin_df)
#     return trips_df
