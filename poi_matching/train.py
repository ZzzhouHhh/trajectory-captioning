import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import joblib

from utils.spatial import add_random_noise_to_location
from .models import POIModel, POIClassifier


# ========== 模型加载/保存 ==========

def load_poi_model(path, device):
    model = POIModel().to(device)
    model.load_state_dict(torch.load(path))
    model.eval()
    print(f'Model loaded from {path}')
    return model


def save_poi_model(model, path):
    torch.save(model.state_dict(), path)
    print(f'Model saved to {path}')


# ========== 训练POI分类器 ==========

def train_poi_classifier(train_loader, model1, poi_model, optimizer, criterion,
                         device, epochs=400):
    """
    训练POI分类器。

    model1: 预训练的intent预测模型（输出9维分布）
    poi_model: 待训练的POI分类器
    训练时：先通过model1获取intent分布，再拼接到特征输入poi_model。
    """
    for epoch in range(epochs):
        poi_model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_predictions = 0

        for batch_features, batch_labels in train_loader:
            batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)

            with torch.no_grad():
                intent_distribution = model1(batch_features).to(device)
            combined_input = torch.cat((batch_features, intent_distribution), dim=1)

            optimizer.zero_grad()
            outputs = poi_model(combined_input)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_predictions += batch_labels.size(0)
            correct_predictions += (predicted == batch_labels).sum().item()

        epoch_accuracy = 100 * correct_predictions / total_predictions if total_predictions > 0 else 0
        if (epoch + 1) % 50 == 0:
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {running_loss / len(train_loader):.4f}, "
                  f"Accuracy: {epoch_accuracy:.2f}%")


# ========== 评估POI分类器 ==========

def evaluate_poi_classifier(test_loader, model1, poi_model, device):
    """
    评估POI分类器的Top-1准确率（中的ACC指标）。
    """
    poi_model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_features, batch_labels in test_loader:
            batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)
            intent_distribution = model1(batch_features).to(device)
            combined_input = torch.cat((batch_features, intent_distribution), dim=1)
            outputs = poi_model(combined_input)
            _, predicted = torch.max(outputs.data, 1)
            total += batch_labels.size(0)
            correct += (predicted == batch_labels).sum().item()
    acc = 100 * correct / total
    print(f"Test Accuracy (ACC): {acc:.2f}%")
    return acc


# ========== 完整训练流程 ==========

def main_train_pipeline(data_path, device='cuda:0'):
    """
    完整训练流程：
    1. 加载签到数据，加GPS随机偏移(noise_radius=100m， d=100)
    2. 训练model1（intent预测器）
    3. 训练poi_classifier（POI匹配）
    4. 保存所有模型和预处理器
    """
    device = torch.device(device if torch.cuda.is_available() else 'cpu')

    # 加载并预处理
    df = pd.read_csv(data_path)
    noise_radius = 100
    df = add_random_noise_to_location(df, noise_radius=noise_radius)

    # 特征和标签
    X = df[['lat', 'lon', 'hour', 'weekday', 'is_weekend']].values
    y = df['classification'].values

    scaler = StandardScaler()
    X[:, [0, 1, 2, 3]] = scaler.fit_transform(X[:, [0, 1, 2, 3]])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(device)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long).to(device)

    # 训练model1
    model1 = POIModel().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model1.parameters(), lr=0.002)

    for epoch in range(1000):
        model1.train()
        outputs = model1(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch + 1}/1000], Loss: {loss.item():.4f}')

    # 测试model1
    model1.eval()
    with torch.no_grad():
        outputs = model1(X_test_tensor)
        _, predicted = torch.max(outputs, 1)
        accuracy = (predicted == y_test_tensor).sum().item() / y_test_tensor.size(0)
        print(f'Model1 Test Accuracy: {accuracy * 100:.2f}%')

    torch.save(model1.state_dict(), 'poi_model_intent.pth')
    joblib.dump(scaler, 'scaler.pkl')

    # 训练POI分类器
    data_full = pd.read_csv(data_path)
    data_full = add_random_noise_to_location(data_full, noise_radius=noise_radius)

    label_encoder = LabelEncoder()
    data_full['poi_id'] = label_encoder.fit_transform(data_full['poi_id'])

    features = data_full[['lat', 'lon', 'hour', 'weekday', 'is_weekend']].values
    labels = data_full['poi_id'].values
    X_train2, X_test2, y_train2, y_test2 = train_test_split(features, labels, test_size=0.2, random_state=36)
    X_train2[:, [0, 1, 2, 3]] = scaler.fit_transform(X_train2[:, [0, 1, 2, 3]])
    X_test2[:, [0, 1, 2, 3]] = scaler.transform(X_test2[:, [0, 1, 2, 3]])

    train_ds = TensorDataset(torch.tensor(X_train2, dtype=torch.float32),
                             torch.tensor(y_train2, dtype=torch.long))
    test_ds = TensorDataset(torch.tensor(X_test2, dtype=torch.float32),
                            torch.tensor(y_test2, dtype=torch.long))
    train_loader = DataLoader(train_ds, batch_size=4096, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=4096, shuffle=False)

    num_intents = 9
    num_pois = len(label_encoder.classes_)
    poi_model = POIClassifier(num_intents, num_pois).to(device)
    opt2 = optim.Adam(poi_model.parameters(), lr=0.001)
    crit2 = nn.CrossEntropyLoss()

    train_poi_classifier(train_loader, model1, poi_model, opt2, crit2, device, epochs=2000)
    evaluate_poi_classifier(test_loader, model1, poi_model, device)

    torch.save(poi_model.state_dict(), "poi_classifier.pth")
    joblib.dump(label_encoder, "label_encoder.pkl")

    return model1, poi_model, scaler, label_encoder
