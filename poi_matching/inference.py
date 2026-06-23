import numpy as np
import torch


def predict_probabilities(model, input_features, device):
    """
    输入纬度、经度、小时、星期几和是否为周末，返回intent概率分布（9维）。
    """
    model.eval()
    with torch.no_grad():
        input_tensor = torch.tensor(input_features, dtype=torch.float32).unsqueeze(0).to(device)
        logits = model(input_tensor)
        probabilities = torch.softmax(logits, dim=1).cpu().numpy()
    return probabilities


def predict_poi_id(lat, lon, hour, weekday, is_weekend, probability_distribution,
                   scaler, poi_model, label_encoder, device):
    """
    对单个轨迹点预测POI ID。

    输入：
    - lat, lon, hour, weekday, is_weekend: 原始特征
    - probability_distribution: model1输出的9维intent分布
    - scaler: 训练时的StandardScaler
    - poi_model: POIClassifier
    - label_encoder: POI id -> index的编码器

    返回：预测的POI ID（原始ID，非编码后的index）
    """
    features = np.array([[lat, lon, hour, weekday, is_weekend]], dtype=np.float32)

    # 标准化前四列
    features[:, [0, 1, 2, 3]] = scaler.transform(features[:, [0, 1, 2, 3]])

    input_tensor = torch.tensor(features, dtype=torch.float32).to(device)
    intent_distribution = torch.tensor(probability_distribution, dtype=torch.float32).to(device)

    combined_input = torch.cat((input_tensor, intent_distribution), dim=1)

    with torch.no_grad():
        output_distribution = torch.softmax(poi_model(combined_input), dim=1)[0]

    predicted_index = torch.argmax(output_distribution).item()
    predicted_poi_id = label_encoder.inverse_transform([predicted_index])[0]
    return predicted_poi_id


# ========== 批量轨迹POI匹配 ==========

def match_poi_to_trips(df, model1, poi_model, scaler, label_encoder, device):
    """
    对轨迹DataFrame中的所有上下车点预测POI。
    返回 (pickup_poi_id, dropoff_poi_id) 列表。
    """
    poi_ids = []

    for _, row in df.iterrows():
        # Pickup点预测
        pickup_probabilities = predict_probabilities(
            model1,
            [[row['pickup_latitude'], row['pickup_longitude'],
              row['pickup_hour'], row['pickup_weekday'], row['pickup_is_weekend']]],
            device
        )[0]
        pickup_poi_id = predict_poi_id(
            row['pickup_latitude'], row['pickup_longitude'],
            row['pickup_hour'], row['pickup_weekday'], row['pickup_is_weekend'],
            pickup_probabilities, scaler, poi_model, label_encoder, device
        )

        # Dropoff点预测
        dropoff_probabilities = predict_probabilities(
            model1,
            [[row['dropoff_latitude'], row['dropoff_longitude'],
              row['dropoff_hour'], row['dropoff_weekday'], row['dropoff_is_weekend']]],
            device
        )[0]
        dropoff_poi_id = predict_poi_id(
            row['dropoff_latitude'], row['dropoff_longitude'],
            row['dropoff_hour'], row['dropoff_weekday'], row['dropoff_is_weekend'],
            dropoff_probabilities, scaler, poi_model, label_encoder, device
        )

        poi_ids.append((pickup_poi_id, dropoff_poi_id))

    return poi_ids
