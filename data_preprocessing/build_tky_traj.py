import pandas as pd
import numpy as np
import random
from datetime import timedelta
from tqdm import tqdm


# ========== 签到数据排序 ==========

def sorted_traj_by_user(filename):
    """按用户ID和时间戳排序签到数据"""
    df = pd.read_csv(filename)
    df['utcTimestamp'] = pd.to_datetime(df['utcTimestamp'])
    df.sort_values(by=['userId', 'utcTimestamp'], inplace=True)
    df.reset_index(drop=True, inplace=True)
    df.to_csv('sorted_checkins_tky.csv', index=False)
    print("排序后的签到文件已生成: sorted_checkins_tky.csv")
    return df


# ========== 从签到数据提取OD轨迹 ==========

def get_traj_from_checkins(df, excluded_categories=None):
    """
    从签到数据中提取OD轨迹对。
    条件：
    - 两个签到点时间差在15~120分钟之间
    - 类别不同
    - 排除交通枢纽类别
    每个用户最多选2对。
    """
    if excluded_categories is None:
        excluded_categories = {
            'Train Station', 'Rail Station',
            'Bus Station', 'Subway Station',
        }

    df['utcTimestamp'] = pd.to_datetime(df['utcTimestamp'])
    df.sort_values(by=['userId', 'utcTimestamp'], inplace=True)

    result = []

    for user_id, group in tqdm(df.groupby('userId'), desc="构建TKY轨迹"):
        checkins = group.to_dict(orient='records')
        valid_pairs = []

        for i in range(len(checkins)):
            for j in range(i + 1, len(checkins)):
                t1 = checkins[i]['utcTimestamp']
                t2 = checkins[j]['utcTimestamp']
                time_diff = (t2 - t1).total_seconds() / 60

                cat1 = checkins[i]['venueCategory']
                cat2 = checkins[j]['venueCategory']

                if cat1 in excluded_categories or cat2 in excluded_categories:
                    continue

                if 15 < time_diff < 120 and cat1 != cat2:
                    valid_pairs.append((checkins[i], checkins[j]))

        selected_pairs = random.sample(valid_pairs, min(2, len(valid_pairs))) if valid_pairs else []

        for origin, destination in selected_pairs:
            result.append({
                'userId': user_id,
                'origin_poi_id': origin['poi_id'],
                'origin_category': origin['venueCategory'],
                'origin_time': origin['utcTimestamp'],
                'origin_latitude': origin['latitude'],
                'origin_longitude': origin['longitude'],
                'dest_poi_id': destination['poi_id'],
                'dest_category': destination['venueCategory'],
                'dest_time': destination['utcTimestamp'],
                'dest_latitude': destination['latitude'],
                'dest_longitude': destination['longitude'],
                'time_diff_minutes': (destination['utcTimestamp'] - origin['utcTimestamp']).total_seconds() / 60
            })

    result_df = pd.DataFrame(result)
    print(f"构建了 {len(result_df)} 条TKY轨迹")
    return result_df


# ========== TKY OD轨迹与POI信息拼接 ==========

def poi_info_match_tky(df_od, df_poi):
    """将TKY OD轨迹数据与POI详细信息拼接"""
    df_od = df_od[['userId', 'origin_poi_id', 'dest_poi_id', 'origin_time', 'dest_time']]

    poi_fields = ['poi_id', 'venueCategory', 'name', 'category']

    df_origin = df_poi[poi_fields].copy()
    df_origin.columns = ['origin_poi_id', 'origin_venueCategory', 'origin_name', 'origin_category']

    df_dest = df_poi[poi_fields].copy()
    df_dest.columns = ['dest_poi_id', 'dest_venueCategory', 'dest_name', 'dest_category']

    df_merged = df_od.merge(df_origin, on='origin_poi_id', how='left')
    df_merged = df_merged.merge(df_dest, on='dest_poi_id', how='left')

    df_merged['pickup_time'] = pd.to_datetime(df_merged['origin_time'])
    df_merged['dropoff_time'] = pd.to_datetime(df_merged['dest_time'])
    df_merged['pickup_hour'] = df_merged['pickup_time'].dt.hour
    df_merged['dropoff_hour'] = df_merged['dropoff_time'].dt.hour
    df_merged['pickup_week'] = df_merged['pickup_time'].dt.day_name()

    np.random.seed(42)
    df_merged['passenger_count'] = np.random.randint(1, 5, size=len(df_merged))

    result_df = pd.DataFrame({
        'trajectory_id': range(1, len(df_merged) + 1),
        'pickup_venueCategory': df_merged['origin_venueCategory'],
        'pickup_name': df_merged['origin_name'],
        'pickup_category': df_merged['origin_category'],
        'dropoff_venueCategory': df_merged['dest_venueCategory'],
        'dropoff_name': df_merged['dest_name'],
        'dropoff_category': df_merged['dest_category'],
        'passenger_count': df_merged['passenger_count'],
        'pickup_hour': df_merged['pickup_hour'],
        'dropoff_hour': df_merged['dropoff_hour'],
        'pickup_week': df_merged['pickup_week']
    })

    return result_df
