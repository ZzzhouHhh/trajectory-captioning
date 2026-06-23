import torch
import torch.nn as nn
import torch.nn.functional as F


# ========== GCN 层  ==========

class GCNLayer(nn.Module):
    """
    单层图卷积: H' = σ(D^{-1/2} A' D^{-1/2} H W)
    其中 A' = A + I (加自环)
    """

    def __init__(self, in_dim, out_dim):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x, adj):
        degree_matrix = torch.diag(adj.sum(1))
        degree_inv_sqrt = torch.inverse(torch.sqrt(degree_matrix))
        normalized_adj = degree_inv_sqrt @ adj @ degree_inv_sqrt
        return F.relu(self.linear(normalized_adj @ x))


# ========== 两层GCN ==========

class GCN(nn.Module):
    """两层图卷积网络"""

    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCN, self).__init__()
        self.gcn1 = GCNLayer(input_dim, hidden_dim)
        self.gcn2 = GCNLayer(hidden_dim, output_dim)

    def forward(self, x, adj):
        x = self.gcn1(x, adj)
        x = self.gcn2(x, adj)
        return x


# ========== 带类别解码器的完整GCN ==========

class TimeCategoryGCN(nn.Module):
    """
    带类别分布预测解码器的GCN。

    前向传播: 公式(14)
    解码器: 从时间节点嵌入 → 子类别概率分布 p̂_t = softmax(W_dec · h_t)
    对应公式(16)。

    训练后的 h_time 作为POI匹配模块的输入特征。
    """

    def __init__(self, input_dim, hidden_dim, output_dim, num_categories):
        super(TimeCategoryGCN, self).__init__()
        self.gcn1 = GCNLayer(input_dim, hidden_dim)
        self.gcn2 = GCNLayer(hidden_dim, output_dim)
        # 解码器：映射时间节点嵌入到子类别分布
        self.decoder = nn.Linear(output_dim, num_categories)

    def forward(self, x, adj):
        x = self.gcn1(x, adj)
        x = self.gcn2(x, adj)
        return x

    def predict_distribution(self, time_node_embeddings):
        """预测子类别概率分布 ()"""
        logits = self.decoder(time_node_embeddings)
        return F.softmax(logits, dim=-1)
