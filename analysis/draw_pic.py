"""
可视化脚本。

包含：
- ω对准确率影响图 
- d对准确率影响图 
- 聚类特征雷达图（附录）
"""

import matplotlib.pyplot as plt
import numpy as np
import os

# 设施名称映射（对应9类POI类别）
FACILITY_NAMES = [
    "Recreation and Culture",    # 0
    "Outdoors and Sightseeing",  # 1
    "Shop and Service",          # 2
    "Restaurant",                # 3
    "Educational Facilities",    # 4
    "Transportation",            # 5
    "Residential Areas",         # 6
    "Medical Facilities",        # 7
    "Office Buildings"           # 8
]


# ========== ω 参数影响图 ==========

def plot_omega_accuracy(omega_values, accuracy_values, output_path='output/w_accuracy.jpg'):
    """
    绘制 ω 对准确率的影响曲线 。

    来自原始 draw_pic/draw_w.py。
    """
    plt.figure(figsize=(8, 5))
    plt.plot(omega_values, accuracy_values, marker='o', linestyle='-',
             color='black', label='Accuracy')

    plt.xlabel('ω', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.title('Effect of Hyperparameter ω on Accuracy', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xticks(omega_values)
    plt.ylim(min(accuracy_values) - 1, max(accuracy_values) + 1)
    plt.legend()
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    print(f"ω effect plot saved to {output_path}")


# ========== d 参数影响图 ==========

def plot_d_accuracy(d_values, accuracy_values, output_path='output/d_accuracy.jpg'):
    """
    绘制偏移距离 d 对准确率的影响曲线 。
    """
    plt.figure(figsize=(8, 5))
    plt.plot(d_values, accuracy_values, marker='s', linestyle='-',
             color='black', label='Accuracy')

    plt.xlabel('Max Offset Distance d (meters)', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.title('Impact of Maximum Offset Distance on Accuracy', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xticks(d_values)
    plt.legend()
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    print(f"d effect plot saved to {output_path}")


# ========== 聚类特征图 ==========

def plot_cluster_features(cluster_id, red_data, blue_data, output_dir='output/cluster_plots'):
    """
    绘制单个聚类的 pickup(红) vs dropoff(蓝) 特征对比图。

    来自原始 trip_purpose/code/draw_pic.py。
    横轴：9类POI设施
    纵轴：归一化参与度
    """
    plt.figure(figsize=(12, 6))

    positions = range(1, 10)  # 横轴位置1-9
    plt.plot(positions, red_data, 'r-o', lw=2, ms=8, label='Pickup Features')
    plt.plot(positions, blue_data, 'b--s', lw=2, ms=8, label='Dropoff Features')

    plt.xticks(positions, FACILITY_NAMES, rotation=45, ha='right',
               fontsize=10, fontweight='demibold')

    plt.title(f'Cluster {cluster_id} Activity Patterns', fontsize=14, pad=20)
    plt.xlabel('Facility Category', fontsize=12, labelpad=15)
    plt.ylabel('Normalized Engagement Level', fontsize=12, labelpad=10)
    plt.grid(True, alpha=0.3, linestyle=':')
    plt.legend(fontsize=10, loc='upper right')
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f'cluster_{cluster_id}_pattern.png'), dpi=150)
    plt.close()


# ========== 批量绘制所有聚类 ==========

def plot_all_clusters(cluster_means_df, output_dir='output/cluster_plots'):
    """
    对所有聚类批量绘制特征图。

    参数:
        cluster_means_df: 聚类特征均值DataFrame
    """
    red_features = [str(i) for i in range(4, 13)]    # 原4-12维(pickup co)
    blue_features = [str(i) for i in range(13, 22)]  # 原13-21维(dropoff co)

    for cluster_id in cluster_means_df.index:
        red_data = cluster_means_df.loc[cluster_id, red_features].values
        blue_data = cluster_means_df.loc[cluster_id, blue_features].values
        plot_cluster_features(cluster_id, red_data, blue_data, output_dir)

    print(f"所有聚类图已保存至 {output_dir}")
