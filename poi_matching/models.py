import torch
import torch.nn as nn


# ========== 基础POI推荐模型（含category嵌入） ==========

class POIRecommenderModel(nn.Module):
    """
    MLP模型：输入(lat, lon, hour, weekday) + category嵌入 -> 预测POI_id。
    """

    def __init__(self, input_dim, num_classes, category_dim, category_embed_dim=8):
        super(POIRecommenderModel, self).__init__()
        self.category_embeddings = nn.Embedding(category_dim, category_embed_dim)
        self.fc1 = nn.Linear(input_dim + category_embed_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, num_classes)

    def forward(self, x, category_ids):
        category_embed = self.category_embeddings(category_ids)
        x = torch.cat((x, category_embed), dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return self.fc4(x)


# ========== Intent预测模型（5特征 -> 9类intent） ==========

class POIModel(nn.Module):
    """
    简单3层MLP：5特征 -> 9类intent分布。

    输入：[lat, lon, hour, weekday, is_weekend]
    输出：9维intent概率分布（对应9种POI类别）
    GSRM的model1（预训练intent预测器）。
    """

    def __init__(self):
        super(POIModel, self).__init__()
        self.fc1 = nn.Linear(5, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 9)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# ========== POI分类器（含intent分布） ==========

class POIClassifier(nn.Module):
    """
    融合intent分布的POI分类器。

    输入：[lat, lon, hour, weekday, is_weekend] + [9维intent分布]
    输出：POI_id logits

    公式(19-21)中 Φ 映射的简化版，
    完整版为 POIClassifierWithContext。
    """

    def __init__(self, num_intents, num_pois):
        super(POIClassifier, self).__init__()
        self.fc1 = nn.Linear(5 + num_intents, 128)
        self.dropout1 = nn.Dropout(0.1)
        self.fc2 = nn.Linear(128, 64)
        self.dropout2 = nn.Dropout(0.1)
        self.fc3 = nn.Linear(64, num_pois)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x


# ========== 完整版POI分类器（含co特征+时间子类别表示） ==========

class POIClassifierWithContext(nn.Module):
    """
    完整版POI匹配模型，GSRM。

    输入 = [raw_features(5), co_features(18), time_subcategory_hidden(64)]
    输出 = P(poi_j | trajectory_point)  ()

    映射 Φ 由多层线性变换+Dropout+ReLU组成 ()。
    最后softmax输出POI概率分布。
    """

    def __init__(self, input_dim, num_pois):
        super(POIClassifierWithContext, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.dropout1 = nn.Dropout(0.1)
        self.fc2 = nn.Linear(256, 128)
        self.dropout2 = nn.Dropout(0.1)
        self.fc3 = nn.Linear(128, 64)
        self.dropout3 = nn.Dropout(0.1)
        self.fc4 = nn.Linear(64, num_pois)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        x = torch.relu(self.fc3(x))
        x = self.dropout3(x)
        x = self.fc4(x)
        return x
