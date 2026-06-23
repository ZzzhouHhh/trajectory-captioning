"""
子类别文本嵌入生成。

使用预训练的 all-MiniLM-L6-v2 模型对POI子类别名称进行编码，
用于时间-子类别图中的 category-category 边相似度计算。

"employs the pre-trained text encoding model MiniLM to
encode subcategory texts and compute the cosine similarity between
subcategory embeddings" ()
"""

import numpy as np
import pandas as pd
import torch
import pickle
from tqdm import tqdm


def init_word_encode_model(model_path, out_dim=8):
    """
    初始化带降维层的SentenceTransformer模型。

    使用 all-MiniLM-L6-v2 (384维) + Dense降维到 out_dim。
    """
    from sentence_transformers import SentenceTransformer, models

    word_embedding_model = models.Transformer(model_path)
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
    dense_model = models.Dense(
        in_features=384, out_features=out_dim,
        activation_function=torch.nn.Tanh()
    )
    model = SentenceTransformer(modules=[word_embedding_model, pooling_model, dense_model])
    return model


def encode_categories(category_list, model):
    """
    使用SentenceTransformer对类别名称列表编码为嵌入向量。

    参数:
        category_list: POI子类别名称列表（如 ['Coffee Shop', 'Pizza Place', ...]）
        model: SentenceTransformer模型

    返回:
        embeddings: numpy array, shape (len(category_list), embed_dim)
    """
    embeddings = model.encode(category_list, convert_to_numpy=True)
    embeddings = np.squeeze(embeddings)
    return embeddings


def generate_category_embeddings(df, model_path, save_path=None, out_dim=384):
    """
    从签到数据中提取所有唯一子类别名称，编码为嵌入向量并保存。

    使用MiniLM编码子类别名称 -> 计算 cosine similarity
                -> 构建 subcategory-subcategory 边。

    参数:
        df: 签到数据DataFrame (需含 venueCategory 列)
        model_path: all-MiniLM-L6-v2 模型路径
        save_path: 嵌入保存路径（pkl）
        out_dim: 输出嵌入维度（默认384, MiniLM原始维度）

    返回:
        category_embeddings: {category_name: embedding_vector} 字典
    """
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_path)

    unique_categories = df['venueCategory'].unique()
    category_embeddings = {}

    for cat in tqdm(unique_categories, desc="Encoding categories"):
        embedding = model.encode([cat], convert_to_numpy=True)
        category_embeddings[cat] = np.squeeze(embedding)

    if save_path:
        with open(save_path, 'wb') as f:
            pickle.dump(category_embeddings, f)
        print(f"类别嵌入已保存至 {save_path}")

    return category_embeddings


def load_embeddings(time_path, category_path):
    """加载预计算的时间和类别嵌入"""
    with open(time_path, 'rb') as f:
        time_embeddings = pickle.load(f)
    with open(category_path, 'rb') as f:
        category_embeddings = pickle.load(f)
    return time_embeddings, category_embeddings
