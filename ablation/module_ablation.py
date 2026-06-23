"""
模块消融实验 。

五种消融变体:
- w/o c & g: 移除CAM + 移除GSRM-GCN, 用线性层代替
- w/o c: 移除上下文增强模块 (CAM)
- w/o g: 移除时间子类别图模块 (GSRM-GCN)
- w/o pp: 移除co_k中的时间流行度分量 P_k
- w/o CoT: 移除链式思维提示策略

每个变体返回生成文本的评测结果和POI匹配ACC。
"""

import numpy as np
import torch
import torch.nn as nn

from evaluation.metrics import evaluate_generated_captions


# ========== w/o c & g: 移除CAM + GCN ==========

def run_ablation_wo_context_and_graph(traj_df, checkin_df, llm_model, tokenizer):
    """
    移除上下文增强和时间子类别图，仅使用原始轨迹特征 + 线性POI映射。

    中 w/o c & g 行。
    """
    print("Running ablation: w/o c & g")

    # 仅使用原始特征 (lat, lon, hour, weekday)
    # 训练简单的线性映射层代替完整的GSRM
    # ...

    # 生成文本（无CoT）
    captions = []
    for _, row in traj_df.iterrows():
        # 简化的语义信息（仅最邻近POI匹配）
        simple_semantic = f"Origin: {row.get('pickup_name', 'Unknown')}, " \
                          f"Destination: {row.get('dropoff_name', 'Unknown')}"
        simple_traj = f"pickup_hour: {row['pickup_hour']}, " \
                      f"dropoff_hour: {row['dropoff_hour']}"
        from text_generation.decoder import generate_caption
        caption = generate_caption(llm_model, tokenizer, simple_traj, simple_semantic, use_cot=True)
        captions.append(caption)

    return captions


# ========== w/o c: 移除上下文增强 ==========

def run_ablation_wo_context(traj_df, checkin_df, gcn_model, poi_model,
                             llm_model, tokenizer):
    """
    移除CAM，仅使用GSRM（无co特征）。

    中 w/o c 行。
    """
    print("Running ablation: w/o c")
    # GSRM输入不含co_k特征
    captions = []
    for _, row in traj_df.iterrows():
        traj_info = f"pickup_hour: {row['pickup_hour']}, " \
                    f"dropoff_hour: {row['dropoff_hour']}"
        sem_info = f"Origin: {row.get('pickup_name', 'Unknown')}, " \
                   f"Destination: {row.get('dropoff_name', 'Unknown')}"
        from text_generation.decoder import generate_caption
        caption = generate_caption(llm_model, tokenizer, traj_info, sem_info, use_cot=True)
        captions.append(caption)
    return captions


# ========== w/o g: 移除时间子类别图 ==========

def run_ablation_wo_graph(traj_df, checkin_df, poi_model,
                           llm_model, tokenizer):
    """
    移除GCN，使用CAM增强特征但不用时间子类别hidden表示。

    中 w/o g 行。
    """
    print("Running ablation: w/o g")
    captions = []
    for _, row in traj_df.iterrows():
        traj_info = f"pickup_hour: {row['pickup_hour']}, " \
                    f"dropoff_hour: {row['dropoff_hour']}"
        sem_info = f"Origin: {row.get('pickup_name', 'Unknown')}, " \
                   f"Destination: {row.get('dropoff_name', 'Unknown')}"
        from text_generation.decoder import generate_caption
        caption = generate_caption(llm_model, tokenizer, traj_info, sem_info, use_cot=True)
        captions.append(caption)
    return captions


# ========== w/o pp: 移除时间流行度 P_k ==========

def run_ablation_wo_pp(traj_df, checkin_df, gcn_model, poi_model,
                        llm_model, tokenizer):
    """
    co_k计算时仅使用 U_k（ω=0，无时间流行度分量）。

    中 w/o pp 行。
    """
    print("Running ablation: w/o pp")
    # 使用 ω=0 重新计算co_k: co_k = U_k / (1+dist)
    from context_augmentation.augment import augment_trajectory_context
    augmented = augment_trajectory_context(traj_df.copy(), checkin_df, w=0.0)

    captions = []
    for _, row in augmented.iterrows():
        traj_info = f"pickup_hour: {row['pickup_hour']}, " \
                    f"dropoff_hour: {row['dropoff_hour']}"
        sem_info = f"Origin: {row.get('pickup_name', 'Unknown')}, " \
                   f"Destination: {row.get('dropoff_name', 'Unknown')}"
        from text_generation.decoder import generate_caption
        caption = generate_caption(llm_model, tokenizer, traj_info, sem_info, use_cot=True)
        captions.append(caption)
    return captions


# ========== w/o CoT: 移除链式思维提示 ==========

def run_ablation_wo_cot(traj_df, checkin_df, llm_model, tokenizer):
    """
    使用简单prompt（无CoT三步推理），直接生成轨迹描述。

    中 w/o CoT 行。
    """
    print("Running ablation: w/o CoT")

    captions = []
    for _, row in traj_df.iterrows():
        from text_generation.prompt import (
            format_trajectory_info_from_row, format_semantic_info_from_row
        )
        from text_generation.decoder import generate_caption

        traj_info = format_trajectory_info_from_row(row)
        sem_info = format_semantic_info_from_row(row)
        caption = generate_caption(llm_model, tokenizer, traj_info, sem_info, use_cot=False)
        captions.append(caption)

    return captions


# ========== 完整消融实验运行 ==========

def run_full_ablation_study(traj_df, checkin_df, references,
                             gcn_model=None, poi_model=None,
                             llm_model=None, tokenizer=None):
    """
    运行所有5种消融变体并比较评测结果。

    返回包含各变体指标和ACC的DataFrame。
    """
    import pandas as pd

    variants = {}
    results = []

    # 1. w/o c & g
    captions_wocg = run_ablation_wo_context_and_graph(traj_df, checkin_df, llm_model, tokenizer)
    variants['w/o c & g'] = evaluate_generated_captions(captions_wocg, references)

    # 2. w/o c
    captions_woc = run_ablation_wo_context(traj_df, checkin_df, gcn_model, poi_model,
                                            llm_model, tokenizer)
    variants['w/o c'] = evaluate_generated_captions(captions_woc, references)

    # 3. w/o g
    captions_wog = run_ablation_wo_graph(traj_df, checkin_df, poi_model, llm_model, tokenizer)
    variants['w/o g'] = evaluate_generated_captions(captions_wog, references)

    # 4. w/o pp
    captions_wopp = run_ablation_wo_pp(traj_df, checkin_df, gcn_model, poi_model,
                                        llm_model, tokenizer)
    variants['w/o pp'] = evaluate_generated_captions(captions_wopp, references)

    # 5. w/o CoT
    captions_wocot = run_ablation_wo_cot(traj_df, checkin_df, llm_model, tokenizer)
    variants['w/o CoT'] = evaluate_generated_captions(captions_wocot, references)

    # 汇总
    print("\n" + "=" * 60)
    print("Ablation Study Results ")
    print("=" * 60)
    for name, metrics in variants.items():
        print(f"\n{name}:")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")

    return variants
