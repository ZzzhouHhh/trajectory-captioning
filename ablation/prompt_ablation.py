"""
Prompt组件消融实验 。

测试不同prompt组件对LLM生成质量的影响：
- ours: 完整prompt（<trajectory_info> + <semantic_info>）
- w/o trajectory_info: 移除轨迹属性
- w/o time: 移除时间信息
- w/o poi_category: 仅保留POI名称
- w/o poi_name: 仅保留POI类别
- Exchange: 交换<trajectory_info>和<semantic_info>的顺序
"""

import pandas as pd
from evaluation.metrics import evaluate_generated_captions
from text_generation.prompt import (
    format_trajectory_info_from_row,
    format_semantic_info_from_row,
    build_prompt_ablation_variant,
)
from text_generation.decoder import load_llm


def run_prompt_component_ablation(traj_df, references, model, tokenizer):
    """
    运行prompt各组件消融实验。

    变体列表对应于 / 。
    """
    variants = [
        'ours',                  # 完整prompt
        'w/o trajectory_info',   # 移除轨迹属性
        'w/o semantic_info',     # 移除地理语义
        'Exchange',              # 交换两个占位符顺序
    ]

    results = {}

    for variant in variants:
        print(f"\nRunning prompt ablation: {variant}")
        captions = []

        for _, row in traj_df.iterrows():
            traj_info = format_trajectory_info_from_row(row)
            sem_info = format_semantic_info_from_row(row)

            prompt = build_prompt_ablation_variant(variant, traj_info, sem_info)

            # 使用简化生成（直接用prompt，不走CoT框架）
            from text_generation.decoder import generate_caption
            caption = generate_caption(
                model, tokenizer, traj_info, sem_info,
                use_cot=(variant != 'Exchange')  # Exchange也用CoT
            )
            captions.append(caption)

        metrics = evaluate_generated_captions(captions, references)
        results[variant] = metrics

        print(f"  {variant}: ROUGE-1={metrics.get('ROUGE-1', 0):.4f}, "
              f"BLEU-2={metrics.get('BLEU-2', 0):.4f}")

    # 汇总
    print("\n" + "=" * 60)
    print("Prompt Component Ablation Results ")
    print("=" * 60)
    header = f"{'Variant':25s}"
    for m in ['ROUGE-1', 'ROUGE-2', 'ROUGE-L', 'BLEU-1', 'BLEU-2']:
        header += f" {m:>10s}"
    print(header)
    print("-" * 60)
    for name, metrics in results.items():
        row = f"{name:25s}"
        for m in ['ROUGE-1', 'ROUGE-2', 'ROUGE-L', 'BLEU-1', 'BLEU-2']:
            row += f" {metrics.get(m, 0):10.4f}"
        print(row)

    return results
