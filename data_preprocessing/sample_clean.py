import pandas as pd


# ========== 轨迹随机采样 ==========

def random_sample_from_csv(file_path, sample_size, output_path=None):
    # 读取CSV文件
    df = pd.read_csv(file_path)

    # 确保样本大小不超过数据框的行数
    if sample_size > len(df):
        print(f"请求的样本大小 {sample_size} 超过了数据集的总行数 {len(df)}。")
        sample_size = len(df)

    # 随机采样
    sampled_df = df.sample(n=sample_size, random_state=42)

    # 如果提供了输出路径，则将采样结果保存到新的CSV文件
    if output_path:
        sampled_df.to_csv(output_path, index=False)
        print(f"已将采样结果保存到 {output_path}")

    return sampled_df


# ========== 轨迹数据清洗 ==========

def clean_and_select_columns(df, columns_to_keep, output_path=None):
    """
    清洗DataFrame，仅保留指定的列。
    删除经纬度为0的无效记录。
    """
    # 检查并保留指定的列
    if not set(columns_to_keep).issubset(df.columns):
        missing_cols = set(columns_to_keep) - set(df.columns)
        raise ValueError(f"指定的列中有缺失: {missing_cols}")

    cleaned_df = df[columns_to_keep]

    # 删除经纬度为0的记录
    cleaned_df = cleaned_df[
        (cleaned_df['pickup_longitude'] != 0) &
        (cleaned_df['pickup_latitude'] != 0) &
        (cleaned_df['dropoff_longitude'] != 0) &
        (cleaned_df['dropoff_latitude'] != 0)
        ]

    if output_path:
        cleaned_df.to_csv(output_path, index=False)
        print(f"已将清洗后的数据保存到 {output_path}")

    return cleaned_df
