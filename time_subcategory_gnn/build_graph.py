import numpy as np
import pandas as pd
import math
import pickle
import networkx as nx
from collections import defaultdict
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity


# ========== 时间-子类别异构图构建  ==========

def build_graph_from_checkins(df, k=5):
    """
    构建时间-子类别异构图。

    节点类型：
    - 时间节点: (weekday, hour) 元组，共 7×24=168 个可能
    - 类别节点: venueCategory 字符串

    边类型：
    - time-category 边: 基于签到频次，权重归一化
      w_{t,c} = log(1+count) / log(1+max_count)  ()
    - category-category 边: 基于类别相似度，取top-k
      使用 cosine_similarity(category_vectors) ()

    参数:
        df: 签到数据 (需含weekday, hour, venueCategory)
        k: 每个类别节点连接最相似的k个其他类别 (默认5)
    """
    G = nx.Graph()
    time_category_edges = defaultdict(int)

    # 构建 time-category 边 ()
    for _, row in tqdm(df.iterrows(), total=df.shape[0],
                       desc="Processing time-category edges"):
        time_key = (int(row['weekday']), int(row['hour']))
        category_key = row['venueCategory']

        G.add_node(time_key, type="time")
        G.add_node(category_key, type="category")

        time_category_edges[(time_key, category_key)] += 1

    # 归一化 time-category 权重
    max_weight = max(time_category_edges.values())
    for (time_key, category_key), count in time_category_edges.items():
        log_weight = math.log(1 + count)
        normalized_weight = log_weight / math.log(1 + max_weight)
        G.add_edge(time_key, category_key, weight=normalized_weight)

    # 计算 category-category 相似度 ()
    category_keys = list(set(df['venueCategory']))
    # 使用随机初始化作为占位；实际应替换为MiniLM编码的嵌入
    # 参见 embedding/category_embed.py 生成真实嵌入
    category_vectors = np.array([np.random.rand(16) for _ in category_keys])
    cosine_sim_matrix = cosine_similarity(category_vectors)

    for i, category_key in enumerate(tqdm(category_keys,
                                           desc="Processing category-category edges")):
        sorted_indices = cosine_sim_matrix[i].argsort()[::-1]
        top_k_indices = sorted_indices[1:k + 1]  # 排除自身
        for j in top_k_indices:
            similar_category = category_keys[j]
            similarity_score = cosine_sim_matrix[i, j]
            G.add_edge(category_key, similar_category, weight=similarity_score)

    print(f"图构建完成: {len([n for n in G.nodes() if isinstance(n, tuple)])} 个时间节点, "
          f"{len([n for n in G.nodes() if isinstance(n, str)])} 个子类别节点")

    return G
