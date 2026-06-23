import numpy as np
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from functools import partial

from utils.spatial import haversine_distance, get_neighboring_grids
from .popularity import calculate_popularity
from .uniqueness import calculate_u_for_point


# ========== 时空上下文 co_k  ==========

def calculate_co(row, classification, checkin_df, checkin_counts, pickup=True, w=0.5):
    """
    计算时空上下文 co_k = (w * P_k + (1-w) * U_k) / (1 + dist)

    其中：
    - P_k: 时间流行度 ()
    - U_k: 类别独特性 ()
    - w (omega): P_k和U_k的平衡权重，默认0.5（在图3中为0.4）
    - dist: 轨迹点到该类别POI平均位置的Haversine距离
    """
    if pickup:
        hour_field = 'pickup_hour'
        grid_field = 'pickup_grid_id'
        lat_field = 'pickup_latitude'
        lon_field = 'pickup_longitude'
    else:
        hour_field = 'drop_hour'
        grid_field = 'dropoff_grid_id'
        lat_field = 'dropoff_latitude'
        lon_field = 'dropoff_longitude'

    grid_id = row[grid_field]
    lat = row[lat_field]
    lon = row[lon_field]

    grids_to_consider = get_neighboring_grids(grid_id)

    # 获取网格内该类别的POI信息，计算平均距离
    pois = checkin_df[(checkin_df['grid_id'].isin(grids_to_consider)) &
                      (checkin_df['classification'] == classification)]

    if pois.empty:
        dist = 0
    else:
        avg_lat = pois['lat'].mean()
        avg_lon = pois['lon'].mean()
        dist = haversine_distance(lat, lon, avg_lat, avg_lon)

    pp_value = calculate_popularity(row, classification, checkin_counts, pickup=pickup)
    u_value = calculate_u_for_point(grid_id, checkin_df).get(classification, 0)

    # 处理P_k无穷大的边界情况
    if pp_value == float('inf'):
        pp_value = 10.0

    co_value = (w * pp_value + (1 - w) * u_value) / (1 + dist)

    return co_value


# ========== 并行计算辅助 ==========

def parallel_calculate_co(trips_df, classification, checkin_df, checkin_counts,
                          pickup=True, w=0.5):
    """并行计算某个分类的co值（使用多进程）"""
    func = partial(calculate_co, classification=classification,
                   checkin_df=checkin_df, checkin_counts=checkin_counts,
                   pickup=pickup, w=w)

    with Pool(processes=cpu_count()) as pool:
        results = list(tqdm(
            pool.imap(func, [row for _, row in trips_df.iterrows()]),
            total=len(trips_df),
            desc=f"Calculating CO Values for Classification {classification}"
        ))

    return results
