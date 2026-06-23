from tqdm import tqdm
from .prompt import (
    format_trajectory_info_from_row, format_poi_info,
    format_semantic_info_from_row,
)
from .decoder import generate_caption_multi_stage, generate_caption_simple


def batch_generate_captions(model, tokenizer, df,
                             use_multi_stage=True,
                             max_new_tokens=128,
                             temperature=0.7, top_p=0.9):
    """
    对轨迹DataFrame批量生成描述。

    参数:
        use_multi_stage: True 使用三阶段推理, False 使用单阶段(w/o MSR)
    """
    captions = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Generating captions"):
        if use_multi_stage:
            traj_info = format_trajectory_info_from_row(row)

            pickup_poi_info = format_poi_info(
                row.get('pickup_name', row.get('pickup_predicted_name', '')),
                row.get('pickup_category', row.get('pickup_predicted_category', '')),
                row.get('pickup_venueCategory', ''),
            )
            dropoff_poi_info = format_poi_info(
                row.get('dropoff_name', row.get('dropoff_predicted_name', '')),
                row.get('dropoff_category', row.get('dropoff_predicted_category', '')),
                row.get('dropoff_venueCategory', ''),
            )

            pickup_hour = row.get('pickup_hour', 0)
            dropoff_hour = row.get('dropoff_hour', 0)
            weekday_map = {0: "Monday", 1: "Tuesday", 2: "Wednesday",
                           3: "Thursday", 4: "Friday", 5: "Saturday", 6: "Sunday"}
            pickup_weekday = row.get('pickup_week', row.get('pickup_weekday', 0))
            if isinstance(pickup_weekday, (int, float)):
                pickup_week = weekday_map.get(int(pickup_weekday) % 7, str(pickup_weekday))
            else:
                pickup_week = str(pickup_weekday)
            dropoff_week = pickup_week  # same trip

            caption = generate_caption_multi_stage(
                model, tokenizer, traj_info,
                pickup_poi_info, dropoff_poi_info,
                pickup_hour, pickup_week,
                dropoff_hour, dropoff_week,
                max_new_tokens=max_new_tokens,
                temperature=temperature, top_p=top_p,
            )
        else:
            traj_info = format_trajectory_info_from_row(row)
            sem_info = format_semantic_info_from_row(row)
            caption = generate_caption_simple(
                model, tokenizer, traj_info, sem_info,
                max_new_tokens=max_new_tokens,
                temperature=temperature, top_p=top_p,
            )

        captions.append(caption)

        if idx < 3:
            print(f"\n--- Caption {idx + 1} ---")
            print(caption)

    df = df.copy()
    df['generated_caption'] = captions
    return df
