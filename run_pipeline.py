"""
端到端轨迹描述生成 pipeline。

整体架构: CAM (上下文增强) -> GSRM (地理语义推理) -> TDM (轨迹解码)

用法:
    python run_pipeline.py --dataset nyc --data_dir ./data --output_dir ./output
"""
import os
import sys
import argparse
import pandas as pd
import numpy as np
import torch
import pickle
import joblib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_preprocessing import (
    random_sample_from_csv, clean_and_select_columns,
    match_nearest_poi, join_poi_info_to_trajectory,
)
from context_augmentation import augment_trajectory_context
from time_subcategory_gnn import (
    build_graph_from_checkins, train_time_subcategory_gcn,
    get_time_subcategory_representation,
)
from poi_matching import (
    POIModel, POIClassifier, POIClassifierWithContext,
    load_poi_model, predict_probabilities, predict_poi_id,
    train_poi_classifier, evaluate_poi_classifier,
)
from text_generation import (
    load_llm, generate_caption,
    format_trajectory_info_from_row, format_semantic_info_from_row,
)


def main():
    parser = argparse.ArgumentParser(description='Trajectory Captioning Pipeline')
    parser.add_argument('--dataset', type=str, default='nyc', choices=['nyc', 'tky'])
    parser.add_argument('--data_dir', type=str, required=True, help='数据目录')
    parser.add_argument('--output_dir', type=str, default='./output', help='输出目录')
    parser.add_argument('--llm_model', type=str,
                        default='meta-llama/Meta-Llama-3.1-8B-Instruct')
    parser.add_argument('--omega', type=float, default=0.4,
                        help='时空上下文权重 (0.4)')
    parser.add_argument('--noise_radius', type=float, default=100.0,
                        help='POI匹配训练的GPS偏移半径(米)')
    parser.add_argument('--num_categories', type=int, default=9)
    parser.add_argument('--gcn_epochs', type=int, default=200)
    parser.add_argument('--poi_epochs', type=int, default=2000)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--skip_gcn', action='store_true')
    parser.add_argument('--skip_poi', action='store_true')
    parser.add_argument('--skip_llm', action='store_true')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    print("=" * 60)
    print("Trajectory Captioning Pipeline")
    print("Synthesizing Geographical Semantics and LLMs")
    print("=" * 60)

    # ==================== Step 1: 数据加载 ====================
    print("\n[Step 1/5] 数据加载与预处理")
    traj_path = os.path.join(args.data_dir, 'nyc_taxi_trips.csv')
    poi_path = os.path.join(args.data_dir, 'poi_info_batch.csv')
    checkin_path = os.path.join(args.data_dir, 'nyc_checkins.csv')

    # 采样并清洗轨迹数据
    sample_df = random_sample_from_csv(traj_path, sample_size=100000)
    columns_to_keep = ['passenger_count', 'tpep_pickup_datetime', 'tpep_dropoff_datetime',
                       'pickup_longitude', 'pickup_latitude',
                       'dropoff_longitude', 'dropoff_latitude', 'total_amount']
    traj_df = clean_and_select_columns(
        sample_df, columns_to_keep,
        output_path=os.path.join(args.output_dir, 'cleaned_trajectories.csv')
    )

    # POI最近邻匹配
    poi_df = pd.read_csv(poi_path)
    traj_df = match_nearest_poi(traj_df, poi_df)
    traj_df.to_csv(os.path.join(args.output_dir, 'trajectory_with_poi.csv'), index=False)

    checkin_df = pd.read_csv(checkin_path)
    print(f"  加载了 {len(traj_df)} 条轨迹, {len(checkin_df)} 条签到记录")

    # ==================== Step 2: 上下文增强 (CAM) ====================
    print("\n[Step 2/5] 轨迹上下文增强 (CAM)")

    traj_df = augment_trajectory_context(
        traj_df, checkin_df,
        num_categories=args.num_categories,
        w=args.omega,
    )
    traj_df.to_csv(os.path.join(args.output_dir, 'augmented_trajectories.csv'), index=False)
    print(f"  上下文增强完成。每条轨迹增加了 {args.num_categories * 2} 维co特征。")

    # ==================== Step 3: 时间子类别GCN (GSRM) ====================
    print("\n[Step 3/5] 时间-子类别图神经网络 (GSRM-GCN)")

    gcn_save_path = os.path.join(args.output_dir, 'gcn_model.pth')

    # 构建时间-子类别异构图
    G = build_graph_from_checkins(checkin_df, k=5)

    model, node_to_id, id_to_node, adj_tensor, node_features = \
        train_time_subcategory_gcn(
            checkin_df, G,
            hidden_dim=128, output_dim=64,
            num_epochs=args.gcn_epochs,
            device=args.device,
        )
    torch.save(model.state_dict(), gcn_save_path)
    print(f"  GCN模型已保存至 {gcn_save_path}")

    # ==================== Step 4: POI匹配 (GSRM-POI) ====================
    print("\n[Step 4/5] POI匹配 (GSRM-POI)")

    from utils import add_random_noise_to_location
    from sklearn.preprocessing import LabelEncoder, StandardScaler

    # 使用签到数据训练POI匹配模型（加GPS随机偏移）
    ck_df = pd.read_csv(checkin_path)
    ck_df = add_random_noise_to_location(ck_df, noise_radius=args.noise_radius)

    # 特征
    X = ck_df[['lat', 'lon', 'hour', 'weekday', 'is_weekend']].values
    y = ck_df['classification'].values

    scaler = StandardScaler()
    X[:, [0, 1, 2, 3]] = scaler.fit_transform(X[:, [0, 1, 2, 3]])

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_t = torch.tensor(y_train, dtype=torch.long).to(device)

    # 训练intent预测模型(model1)
    model1 = POIModel().to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model1.parameters(), lr=0.002)

    for epoch in range(1000):
        model1.train()
        outputs = model1(X_train_t)
        loss = criterion(outputs, y_train_t)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 200 == 0:
            print(f'  Model1 Epoch [{epoch + 1}/1000], Loss: {loss.item():.4f}')

    # 在测试集评估
    X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test_t = torch.tensor(y_test, dtype=torch.long).to(device)
    model1.eval()
    with torch.no_grad():
        outputs = model1(X_test_t)
        _, predicted = torch.max(outputs, 1)
        acc = (predicted == y_test_t).sum().item() / y_test_t.size(0)
        print(f'  Model1 Test Accuracy: {acc * 100:.2f}%')

    torch.save(model1.state_dict(), os.path.join(args.output_dir, 'model1_intent.pth'))
    joblib.dump(scaler, os.path.join(args.output_dir, 'scaler.pkl'))

    # 训练POI分类器
    label_encoder = LabelEncoder()
    ck_df['poi_id'] = label_encoder.fit_transform(ck_df['poi_id'])

    features = ck_df[['lat', 'lon', 'hour', 'weekday', 'is_weekend']].values
    labels = ck_df['poi_id'].values
    X_train2, X_test2, y_train2, y_test2 = train_test_split(features, labels, test_size=0.2, random_state=36)
    X_train2[:, [0, 1, 2, 3]] = scaler.fit_transform(X_train2[:, [0, 1, 2, 3]])
    X_test2[:, [0, 1, 2, 3]] = scaler.transform(X_test2[:, [0, 1, 2, 3]])

    from torch.utils.data import DataLoader, TensorDataset
    train_ds = TensorDataset(torch.tensor(X_train2, dtype=torch.float32),
                             torch.tensor(y_train2, dtype=torch.long))
    test_ds = TensorDataset(torch.tensor(X_test2, dtype=torch.float32),
                            torch.tensor(y_test2, dtype=torch.long))
    train_loader = DataLoader(train_ds, batch_size=4096, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=4096, shuffle=False)

    num_intents = args.num_categories
    num_pois = len(label_encoder.classes_)
    poi_model = POIClassifier(num_intents, num_pois).to(device)
    opt2 = torch.optim.Adam(poi_model.parameters(), lr=0.001)
    crit2 = torch.nn.CrossEntropyLoss()

    print(f"  Training POI classifier ({num_pois} POIs)...")
    train_poi_classifier(train_loader, model1, poi_model, opt2, crit2, device, epochs=args.poi_epochs)
    acc_final = evaluate_poi_classifier(test_loader, model1, poi_model, device)

    torch.save(poi_model.state_dict(), os.path.join(args.output_dir, 'poi_classifier.pth'))
    joblib.dump(label_encoder, os.path.join(args.output_dir, 'label_encoder.pkl'))

    # ==================== Step 5: 文本生成 (TDM) ====================
    print("\n[Step 5/5] 轨迹解码与描述生成 (TDM)")

    if args.skip_llm:
        print("  跳过LLM文本生成 (--skip_llm).")
        print(f"\n{'=' * 60}")
        print(f"Pipeline 完成! POI匹配准确率 (ACC): {acc_final:.2f}%")
        print(f"输出文件保存在: {args.output_dir}")
        return

    # 加载LLM
    llm, tokenizer = load_llm(args.llm_model, device=args.device)

    # 对每条轨迹生成描述
    captions = []
    for idx, row in traj_df.iterrows():
        traj_info = format_trajectory_info_from_row(row)
        sem_info = format_semantic_info_from_row(row)

        caption = generate_caption(
            llm, tokenizer, traj_info, sem_info,
            use_cot=True, max_new_tokens=128, temperature=0.7, top_p=0.9
        )
        captions.append(caption)

        if idx < 3:
            print(f"\n--- 轨迹描述 {idx + 1} ---")
            print(caption)

    # 保存结果
    import json
    with open(os.path.join(args.output_dir, 'generated_captions.json'), 'w', encoding='utf-8') as f:
        json.dump(captions, f, ensure_ascii=False, indent=2)

    print(f"\n{'=' * 60}")
    print(f"Pipeline 完成! 生成了 {len(captions)} 条轨迹描述。")
    print(f"POI匹配准确率 (ACC): {acc_final:.2f}%")
    print(f"输出文件保存在: {args.output_dir}")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
