import numpy as np
from utils.spatial import get_neighboring_grids


# ========== 类别独特性 U_k  ==========

def calculate_u_for_point(grid_id, checkin_df):
    """
    计算区域内各类别POI的独特性 U_k。

    U_k = -log2((1 + N_k) / (1 + N_total))

    其中 N_k 为区域内第k类POI的唯一数量，
    N_total 为区域内所有POI的总数。
    """
    grids_to_consider = get_neighboring_grids(grid_id)

    unique_poi_counts = checkin_df[checkin_df['grid_id'].isin(grids_to_consider)] \
        .groupby('classification')['poi_id'].nunique()
    total_sum = unique_poi_counts.sum()

    u_values = {}
    for classification, N_k in unique_poi_counts.items():
        u_k = -np.log2((1 + N_k) / (1 + total_sum))
        u_values[classification] = u_k

    return u_values
