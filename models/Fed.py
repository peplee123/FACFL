#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
from scipy.spatial.distance import jensenshannon
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.cluster.hierarchy import linkage, fcluster
import copy
import torch
import math
from networkx.algorithms.community import greedy_modularity_communities
import networkx as nx
from sklearn.cluster import KMeans
import numpy as np
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler
from kmodes.kmodes import KModes
from scipy.stats import entropy
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import SpectralClustering
from sklearn.metrics import pairwise_distances
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score


def FedAvg(w):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg

def wasserstein_distance(p, q):
    epsilon = 1e-10
    p = torch.FloatTensor(p)
    q = torch.FloatTensor(q)
    cdf_p = torch.cumsum(p + epsilon, dim=0)
    cdf_q = torch.cumsum(q + epsilon, dim=0)
    return torch.nn.functional.pairwise_distance(cdf_p.unsqueeze(1), cdf_q.unsqueeze(1), p=1)[0].item()


def kl_divergence(p, q):
    p = np.clip(p, 1e-10, 1.0)
    q = np.clip(q, 1e-10, 1.0)
    return np.sum(p * np.log(p / q))


def euclidean_distance(p, q):
    return np.sqrt(np.sum((p - q) ** 2))


def wasserstein_distance(p, q):
    p = torch.FloatTensor(p)
    q = torch.FloatTensor(q)
    cdf_p = torch.cumsum(p, dim=0)
    cdf_q = torch.cumsum(q, dim=0)
    cdf_q = cdf_q.unsqueeze(1)
    cdf_p = cdf_p.unsqueeze(1)
    return torch.nn.functional.pairwise_distance(cdf_p, cdf_q, p=1)[0].item()


def wasserstein_kmeans(data, n_clusters):
    # 将数据转换为PyTorch张量
    data = torch.FloatTensor(data)

    # 初始化K-means算法
    kmeans = KMeans(n_clusters=n_clusters)

    # 将K-means算法的距离度量设置为Wasserstein距离
    kmeans._distance_func = lambda x, y: wasserstein_distance(x, y)

    # 执行K-means聚类
    kmeans.fit(data)

    # 返回聚类结果
    return kmeans.labels_


def js_divergence(p, q, epsilon=1e-10):
    # Ensure the distributions are normalized
    p = p / np.sum(p)
    q = q / np.sum(q)

    # Add a small constant for numerical stability
    p = p + epsilon
    q = q + epsilon

    # Normalize them again after adding epsilon
    p = p / np.sum(p)
    q = q / np.sum(q)

    m = 0.5 * (p + q)

    return 0.5 * (entropy(p, m, base=2) + entropy(q, m, base=2))




def weight_flatten(model):
    params = []
    for u in model.parameters():
        params.append(u.view(-1))
    params = torch.cat(params)

    return params


def FimBA(w_locals, fim_local, client_distributed, maxcluster,args):
    print('=======================')
    import os
    import time
    output_dir="output_images"
    # 确保保存图像的文件夹存在，如果不存在则创建
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if maxcluster == 1:
        w_avg = copy.deepcopy(w_locals[0])
        for k in w_avg.keys():
            for i in range(1, len(w_locals)):
                w_avg[k] += w_locals[i][k]
            w_avg[k] = torch.div(w_avg[k], len(w_locals))
        w_global_dict = {0: w_avg}
        index_dict = {0: list(range(len(w_locals)))}
        print("Simple FedAvg applied as maxcluster is 1")
        return w_global_dict, index_dict

    print('进入到这')
    # 计算各客户端之间的余弦相似度矩阵
    max_len = max([len(fim) for fim in fim_local])  # 找到最长的向量长度
    fim_local_padded = [np.pad(fim, (0, max_len - len(fim)), 'constant') for fim in fim_local]  # 零填充
    cosine_matrix = cosine_similarity(fim_local_padded)  # 计算余弦相似度矩阵
    # 将相似度矩阵转换为非负距离矩阵
    distance_matrix = np.maximum(0, 1 - cosine_matrix)

    # 设置对角线元素为零
    np.fill_diagonal(distance_matrix, 0)

    # 将相似度矩阵转换为距离矩阵
    # distance_matrix = 1 - cosine_matrix


    # 进行层次聚类
    Z = linkage(distance_matrix, method='average')
    # 选择聚类范围
    cluster_range = range(2, maxcluster+1)  # 设置要测试的聚类数目范围，比如 2 到 10

    # 计算不同聚类数目的轮廓系数，寻找最佳聚类数
    best_num_clusters = 2  # 初始化聚类数目
    best_silhouette_score = -1  # 初始化轮廓系数

    #
    # for num_clusters in range(2, len(fim_local) + 1):
    #     # 根据当前聚类数目生成聚类标签
    #     labels = fcluster(Z, num_clusters, criterion='maxclust')
    #
    #     # 计算轮廓系数
    #     silhouette_avg = silhouette_score(distance_matrix, labels, metric='precomputed')
    #
    #     # 更新最佳聚类数和轮廓系数
    #     if silhouette_avg > best_silhouette_score:
    #         best_silhouette_score = silhouette_avg
    #         best_num_clusters = num_clusters
    #
    # print(f"最佳聚类数目: {best_num_clusters}")
    # print(f"最佳轮廓系数: {best_silhouette_score}")



    # # 保存 WCSS 值
    # wcss = []
    # for n_clusters in cluster_range:
    #     # 基于 n_clusters 聚类
    #     labels = fcluster(Z, n_clusters, 'maxclust')
    #
    #     # 初始化簇中心
    #     cluster_centers = np.zeros((n_clusters, fim_local_padded[0].shape[0]))  # 特征数与向量长度相同
    #
    #     # 计算每个簇的中心
    #     for cluster_label in range(1, n_clusters + 1):
    #         cluster_data = np.array(fim_local_padded)[labels == cluster_label]
    #         cluster_centers[cluster_label - 1] = np.mean(cluster_data, axis=0)
    #
    #     # 计算每个样本到簇中心的距离
    #     cluster_distances = np.zeros(len(fim_local_padded))
    #     for i in range(len(fim_local_padded)):
    #         cluster_label = labels[i]
    #         cluster_center = cluster_centers[cluster_label - 1]
    #         cluster_distances[i] = np.sum((fim_local_padded[i] - cluster_center) ** 2)
    #
    #     # 计算 WCSS 并保存
    #     wcss.append(np.sum(cluster_distances))
    #
    # # 使用拐点法计算最优簇数
    # diff = np.diff(wcss)  # 计算 WCSS 差分
    # second_derivative = np.diff(diff)  # 计算二阶导数
    #
    # # 找到二阶导数首次变为非负的点
    # knee = np.where(second_derivative >= 0)[0][0] + 2  # 加 2 是因为 diff 减少了两个长度
    # optimal_num_clusters = cluster_range[knee]
    # print("Optimal number of clusters:", optimal_num_clusters)
    #
    # # 根据最优聚类数进行聚类
    # labels = fcluster(Z, optimal_num_clusters, 'maxclust')
    # print("Cluster assignments:", labels)
    #
    # # 生成每个簇的索引字典
    # index_dict = {}
    # for i in range(len(labels)):
    #     if labels[i] not in index_dict:
    #         index_dict[labels[i]] = []
    #     index_dict[labels[i]].append(i)
    #
    # # 输出每个簇的索引字典
    # print("Cluster index dictionary:", index_dict)

    best_num_clusters=10
    optimal_labels = fcluster(Z, best_num_clusters, criterion='maxclust')
    print("Cluster assignments:", optimal_labels)
    # 生成每个簇的索引字典
    index_dict = {}
    for i in range(len(optimal_labels)):
        if optimal_labels[i] not in index_dict:
            index_dict[optimal_labels[i]] = []
        index_dict[optimal_labels[i]].append(i)

    # 输出每个簇的索引字典
    print("Cluster index dictionary:", index_dict)
    # # ---- 可视化聚类结果：t-SNE降维 ----
    # # 使用t-SNE将fim_local的高维数据降维到2D
    # fim_local_padded = np.array(fim_local_padded)  # 转换为numpy数组
    # tsne = TSNE(n_components=2, random_state=42)
    # fim_local_2d = tsne.fit_transform(fim_local_padded)
    #
    # # ---- 生成并保存每一轮的图像 ----
    # # 获取当前时间戳，确保文件名唯一
    # timestamp = time.strftime("%Y%m%d-%H%M%S", time.localtime())
    # plot_filename = os.path.join(output_dir, f"tsne_cluster_{timestamp}.png")
    #
    # # 绘制t-SNE结果的散点图
    # plt.figure(figsize=(10, 8))
    # sns.scatterplot(x=fim_local_2d[:, 0], y=fim_local_2d[:, 1], hue=optimal_labels, palette="tab10", s=100, marker='o',
    #                 edgecolor='k')
    # plt.title(f"t-SNE visualization of clustering (Best num clusters = {best_num_clusters})")
    # plt.xlabel("t-SNE component 1")
    # plt.ylabel("t-SNE component 2")
    # plt.legend(title="Cluster", loc='upper right')
    #
    # # 保存图像到指定路径
    # plt.savefig(plot_filename)
    # plt.close()  # 关闭当前图像，防止内存泄漏
    #
    # print(f"Saved t-SNE plot to: {plot_filename}")

    '''
    '''
    # 创建字典来存储每个簇的全局模型
    w_global_dict = {}

    # 根据客户端的fisher信息矩阵的迹计算权重
    trace_sums = [fim.sum().item() for fim in fim_local]  # 计算每个客户端的迹的和

    # 创建全 0 张量，并将其赋值给 w_avg
    for j in index_dict.keys():

        # 创建全 0 张量，并将其赋值给 w_avg
        w_avg = {}
        if args.juhe == 'fisher':
            # 计算当前簇内客户端的迹的总和
            current_cluster_trace_sum = sum([trace_sums[i] for i in index_dict[j]])

            for key, value in w_locals[0].items():
                w_avg[key] = torch.zeros_like(value)

            # 聚合时根据当前簇内客户端的迹的和来分配权重
            for i in index_dict[j]:
                weight = trace_sums[i] / current_cluster_trace_sum  # 按照当前簇内的迹的和计算权重
                for k in w_avg.keys():
                    w_avg[k] += weight * w_locals[i][k]
        elif args.juhe == 'avg':
            # 改 每个簇的模型都得先置0
            for key, value in w_locals[0].items():
                w_avg[key] = torch.zeros_like(value)
            for k in w_avg.keys():
                for i in index_dict[j]:
                    w_avg[k] += w_locals[i][k]
                w_avg[k] = torch.div(w_avg[k], len(index_dict[j]))
        else:
            print('请输入聚合参数')
        # 将当前簇的聚合模型赋值给全局模型字典
        w_global_dict[j] = copy.deepcopy(w_avg)

    return w_global_dict, index_dict


def NewFedBa(w_locals, client_distributed,maxcluster):
    # 将 client_distributed 转换为张量
    client_distributed = [item for item in client_distributed]
    client_distributed = torch.tensor(np.array([item.numpy() for item in client_distributed]))
    # client_distributed = torch.tensor([item.numpy() for item in client_distributed])
    client_distributed = client_distributed.cpu()
    # 将 client_distributed 转换为 NumPy 数组
    data = client_distributed.numpy()
    # print('data',data)
    # 计算样本之间的 Jensen-Shannon 距离
    dist_matrix = np.zeros((data.shape[0], data.shape[0]))
    for i in range(data.shape[0]):
        for j in range(i + 1, data.shape[0]):
            dist = euclidean_distance(data[i], data[j])
            dist_matrix[i, j] = dist_matrix[j, i] = dist

    # 执行层次聚类
    # Z = linkage(dist_matrix, method='ward')
    Z = linkage(dist_matrix, method='average')

    # 计算簇内平方和 (WCSS)
    wcss = []
    cluster_range = range(2, min(maxcluster, data.shape[0] + 1))  # Limit the range to a maximum of 10 clusters
    print('maxcluster',maxcluster)
    for n_clusters in cluster_range:
        labels = fcluster(Z, n_clusters, 'maxclust')
        cluster_centers = np.zeros((n_clusters, data.shape[1]))
        for cluster_label in range(1, n_clusters + 1):
            cluster_data = data[labels == cluster_label]
            cluster_centers[cluster_label - 1] = np.mean(cluster_data, axis=0)
        cluster_distances = np.zeros(data.shape[0])
        for i in range(data.shape[0]):
            cluster_label = labels[i]
            cluster_center = cluster_centers[cluster_label - 1]
            cluster_distances[i] = np.sum((data[i] - cluster_center) ** 2)
        wcss.append(np.sum(cluster_distances))

    # 使用拐点法计算最优簇数
    diff = np.diff(wcss)
    knee = np.argmax(diff) + 1
    optimal_num_clusters = cluster_range[knee]
    print("Optimal number of clusters:", optimal_num_clusters)

    labels = fcluster(Z, optimal_num_clusters, 'maxclust')
    print("Cluster assignments:", labels)
    index_dict = {}
    for i in range(len(labels)):
        if labels[i] not in index_dict:
            index_dict[labels[i]] = []
        index_dict[labels[i]].append(i)
    # print(index_dict)
    # 创建字典来存储每个簇的全局模型
    w_global_dict = {}

    # w_global_avg = FedAvg(w_locals)

    # 创建全 0 张量，并将其赋值给 w_avg
    w_avg = {}
    # for key, value in w_locals[0].items():
    for j in index_dict.keys():
        # 改 每个簇的模型都得先置0
        for key, value in w_locals[0].items():
            w_avg[key] = torch.zeros_like(value)
        for k in w_avg.keys():
            for i in index_dict[j]:
                w_avg[k] += w_locals[i][k]
            w_avg[k] = torch.div(w_avg[k], len(index_dict[j]))
        w_global_dict[j] = copy.deepcopy(w_avg)
    # print('多个全局模型字典长度',len(w_global_dict),'全局模型聚合索引',index_dict)
    return w_global_dict, index_dict

def diff(w_locals, modeldiff_local, maxcluster):
    print('=======================')
    # 计算邻接矩阵（欧氏距离）
    num_clients = len(modeldiff_local)
    adjacency_matrix = np.zeros((num_clients, num_clients))

    for i in range(num_clients):
        for j in range(num_clients):
            adjacency_matrix[i, j] = abs(modeldiff_local[i] - modeldiff_local[j])

    '''
    import matplotlib.pyplot as plt
    import seaborn as sns
    plt.figure(figsize=(10, 8))
    sns.heatmap(distance_matrix, cmap="magma", square=True)
    plt.xlabel("Client ID")
    plt.ylabel("Client ID")
    # plt.title("Heatmap of Cosine Similarity between Clients")
    plt.show()
    plt.savefig("distance_matrix_heatmap.jpg")

    '''

    # 进行层次聚类
    Z = linkage(adjacency_matrix, method='average')
    # 选择聚类范围
    cluster_range = range(2, maxcluster+1)  # 设置要测试的聚类数目范围，比如 2 到 10
    # modeldiff_local = np.array(modeldiff_local)
    # 保存 WCSS 值
    wcss = []
    for n_clusters in cluster_range:
        # 基于 n_clusters 聚类
        labels = fcluster(Z, n_clusters, 'maxclust')

        # 初始化簇中心
        cluster_centers = np.zeros(n_clusters)
        # cluster_centers = np.zeros((n_clusters, modeldiff_local.shape[0]))  # 特征数与向量长度相同
        print('cluster_centers', cluster_centers)
        # 计算每个簇的中心
        for cluster_label in range(1, n_clusters + 1):
            cluster_data = np.array(modeldiff_local)[labels == cluster_label]
            cluster_centers[cluster_label - 1] = np.mean(cluster_data, axis=0)

        # 计算每个样本到簇中心的距离
        cluster_distances = np.zeros(len(modeldiff_local))
        for i in range(len(modeldiff_local)):
            cluster_label = labels[i]
            cluster_center = cluster_centers[cluster_label - 1]
            cluster_distances[i] = np.sum((modeldiff_local[i] - cluster_center) ** 2)

        # 计算 WCSS 并保存
        wcss.append(np.sum(cluster_distances))
        print(wcss)
    # 使用拐点法计算最优簇数
    diff = np.diff(wcss)  # 计算 WCSS 差分
    second_derivative = np.diff(diff)  # 计算二阶导数

    # 找到二阶导数首次变为非负的点
    knee = np.where(second_derivative >= 0)[0][0] + 2  # 加 2 是因为 diff 减少了两个长度
    optimal_num_clusters = cluster_range[knee]
    print("Optimal number of clusters:", optimal_num_clusters)

    # 根据最优聚类数进行聚类
    labels = fcluster(Z, optimal_num_clusters, 'maxclust')
    print("Cluster assignments:", labels)

    # 生成每个簇的索引字典
    index_dict = {}
    for i in range(len(labels)):
        if labels[i] not in index_dict:
            index_dict[labels[i]] = []
        index_dict[labels[i]].append(i)

    # 输出每个簇的索引字典
    print("Cluster index dictionary:", index_dict)

    # 创建字典来存储每个簇的全局模型
    w_global_dict = {}

    # 创建全 0 张量，并将其赋值给 w_avg
    for j in index_dict.keys():
        # 创建全 0 张量，并将其赋值给 w_avg
        w_avg = {}

        # 改 每个簇的模型都得先置0
        for key, value in w_locals[0].items():
            w_avg[key] = torch.zeros_like(value)
        for k in w_avg.keys():
            for i in index_dict[j]:
                w_avg[k] += w_locals[i][k]
            w_avg[k] = torch.div(w_avg[k], len(index_dict[j]))


        # 将当前簇的聚合模型赋值给全局模型字典
        w_global_dict[j] = copy.deepcopy(w_avg)




    return w_global_dict, index_dict
