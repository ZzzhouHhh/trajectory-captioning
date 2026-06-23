import numpy as np
from utils.spatial import get_neighboring_grids


# ========== 时间流行度 P_k  ==========

def calculate_popularity(row, classification, checkin_counts, pickup=True, epsilon=1e-10):
    """
    计算时间窗口内的类别流行度 P_k。

    对于 pickup 点: 时间窗口 = [hour-2, hour]
    对于 dropoff 点: 时间窗口 = [hour, hour+2]

    P_k = -log2(1 - N_{k,T} / N_T)   if 0 < N_{k,T} < N_T
          0                          if N_T == 0
          inf                        if N_{k,T} == N_T (该类独占)
    """
    if pickup:
        hour_field = 'pickup_hour'
        grid_field = 'pickup_grid_id'
        time_window = (-2, 0)
    else:
        hour_field = 'drop_hour'
        grid_field = 'dropoff_grid_id'
        time_window = (0, 2)

    grid_id = row[grid_field]
    pickup_hour = row[hour_field]

    grids_to_consider = get_neighboring_grids(grid_id)

    count_k = checkin_counts[(checkin_counts['grid_id'].isin(grids_to_consider)) &
                             (checkin_counts['classification'] == classification) &
                             (checkin_counts['hour'] >= pickup_hour + time_window[0]) &
                             (checkin_counts['hour'] <= pickup_hour + time_window[1])]['count'].sum()

    total_count = checkin_counts[(checkin_counts['grid_id'].isin(grids_to_consider)) &
                                 (checkin_counts['hour'] >= pickup_hour + time_window[0]) &
                                 (checkin_counts['hour'] <= pickup_hour + time_window[1])]['count'].sum()

    if total_count == 0 or count_k == total_count:
        return 0 if total_count == 0 else np.inf

    ratio = count_k / (total_count + epsilon)
    popularity = -np.log2(1 - ratio)
    return popularity
