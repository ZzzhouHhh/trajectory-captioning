import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import defaultdict

from utils.temporal import time_encoding
from .gcn_model import TimeCategoryGCN


# ========== 损失函数 ==========

def graph_structure_loss(time_embeddings, cat_embeddings, adjacency_list,
                         node_to_id, time_nodes):
    """
    图结构损失 :

    L_graph = Σ_t || h_t - Σ_{c∈N(t)} w_{t,c}·h_c / Σw ||²

    鼓励时间节点嵌入接近其邻居类别节点嵌入的加权聚合。
    """
    device = time_embeddings.device
    loss = torch.tensor(0.0, device=device)
    n_time = len(time_nodes)
    count = 0

    for t in time_nodes:
        if t not in node_to_id or t not in adjacency_list:
            continue
        tid = node_to_id[t]
        neighbors = [(c, w) for c, w in adjacency_list[t] if c in node_to_id]
        if not neighbors:
            continue

        nbr_embeddings = torch.stack([
            cat_embeddings[node_to_id[n] - n_time]
            for n, _ in neighbors
        ])
        weights = torch.tensor([w for _, w in neighbors],
                               dtype=torch.float32, device=device)
        weights = weights / weights.sum()

        target = (weights.unsqueeze(1) * nbr_embeddings).sum(dim=0)
        loss += torch.norm(time_embeddings[tid] - target, p=2) ** 2
        count += 1

    return loss / max(count, 1)


def distribution_loss(time_embeddings, decoder, true_distributions):
    """
    类别分布损失 :

    L_dist = -Σ_t Σ_c y_{t,c} · log(p̂_{t,c})

    交叉熵损失，使预测的子类别分布与签到数据中的真实分布一致。
    """
    logits = decoder(time_embeddings)
    return F.cross_entropy(logits, true_distributions)


def compute_true_distributions(df, time_nodes, n_cat, cat_to_idx):
    """
    从签到数据计算每个时间节点的真实子类别分布 y_t。

    y_{t,c} = count(t, c) / Σ_c' count(t, c')
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    n_time = len(time_nodes)
    dist = torch.zeros(n_time, n_cat, device=device)

    for i, (weekday, hour) in enumerate(time_nodes):
        mask = (df['weekday'] == weekday) & (df['hour'] == hour)
        cats = df.loc[mask, 'venueCategory']
        for cat in cats:
            if cat in cat_to_idx:
                dist[i, cat_to_idx[cat]] += 1
        total = dist[i].sum()
        if total > 0:
            dist[i] /= total

    return dist


# ========== 训练流程 ==========

def train_time_subcategory_gcn(
    df, G,
    hidden_dim=128,
    output_dim=64,
    num_epochs=200,
    lambda_dist=1.0,
    lr=0.01,
    device='cuda:0' if torch.cuda.is_available() else 'cpu'
):
    """
    训练时间-子类别GCN。

    联合损失 (): L = L_graph + λ · L_dist

    返回训练好的模型及图相关数据。
    """
    device = torch.device(device)

    # 创建节点映射
    node_to_id = {node: i for i, node in enumerate(G.nodes())}
    id_to_node = {i: node for node, i in node_to_id.items()}

    time_nodes = [n for n in G.nodes() if isinstance(n, tuple)]
    cat_nodes = [n for n in G.nodes() if isinstance(n, str)]
    n_time = len(time_nodes)
    n_cat = len(cat_nodes)
    n_total = len(G.nodes())

    # 构造邻接矩阵 + 对称归一化
    adj_matrix = np.zeros((n_total, n_total), dtype=np.float32)
    for u, v, data in G.edges(data=True):
        adj_matrix[node_to_id[u], node_to_id[v]] = data["weight"]
        adj_matrix[node_to_id[v], node_to_id[u]] = data["weight"]

    adj_matrix = adj_matrix + np.eye(n_total)
    deg = adj_matrix.sum(axis=1)
    deg_inv_sqrt = np.diag(1.0 / np.sqrt(np.maximum(deg, 1e-12)))
    adj_norm = deg_inv_sqrt @ adj_matrix @ deg_inv_sqrt
    adj_tensor = torch.tensor(adj_norm, dtype=torch.float32, device=device)

    # 初始化节点特征
    feature_dim = 16
    node_features = torch.randn((n_total, feature_dim), dtype=torch.float32, device=device)
    for tp in time_nodes:
        tid = node_to_id[tp]
        te = time_encoding(tp[0], tp[1])[:feature_dim]
        node_features[tid] = torch.tensor(te, dtype=torch.float32)

    # 构建邻接表
    adjacency_list = defaultdict(list)
    for u, v, data in G.edges(data=True):
        adjacency_list[u].append((v, data["weight"]))
        adjacency_list[v].append((u, data["weight"]))

    # 真实类别分布
    cat_to_idx = {cat: i for i, cat in enumerate(cat_nodes)}
    true_dist = compute_true_distributions(df, time_nodes, n_cat, cat_to_idx)

    # 创建模型
    model = TimeCategoryGCN(feature_dim, hidden_dim, output_dim, n_cat).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()

        all_embeddings = model(node_features, adj_tensor)
        time_emb = all_embeddings[:n_time]
        cat_emb = all_embeddings[n_time:]

        L_graph = graph_structure_loss(time_emb, cat_emb, adjacency_list,
                                       node_to_id, time_nodes)
        L_dist = distribution_loss(time_emb, model.decoder, true_dist)

        total_loss = L_graph + lambda_dist * L_dist
        total_loss.backward()
        optimizer.step()

        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch + 1}/{num_epochs}, "
                  f"L_graph={L_graph.item():.4f}, L_dist={L_dist.item():.4f}")

    print(f"训练完成。final_loss={total_loss.item():.4f}")
    return model, node_to_id, id_to_node, adj_tensor, node_features


# ========== 推理 ==========

def get_time_subcategory_representation(model, node_features, adj_tensor,
                                         node_to_id, weekday, hour,
                                         device='cuda:0'):
    """
    获取指定时间(weekday, hour)的GCN隐藏层表示 h_time。
    该向量编码了该时间的子类别分布信息，
    作为POI匹配模块的输入特征 。
    """
    time_key = (weekday, hour)
    if time_key not in node_to_id:
        return torch.zeros(model.gcn2.linear.out_features)

    model.eval()
    with torch.no_grad():
        all_embeddings = model(node_features, adj_tensor)
        tid = node_to_id[time_key]
        return all_embeddings[tid].cpu()
