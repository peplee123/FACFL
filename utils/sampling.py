#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import random
import numpy as np
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split


batch_size = 10
train_size = .8 # merge original training set and test set, then split it manually.
least_samples = batch_size / (1-train_size) # least samples for each client


def separate_data(data, num_clients, num_classes, niid=False, balance=False, partition=None, class_per_client=3, alpha=0.1):
    X = {i: [] for i in range(num_clients)}
    y = {i: [] for i in range(num_clients)}
    statistic = [[] for _ in range(num_clients)]

    dataset_content, dataset_label = data

    dataidx_map = {}

    if not niid:
        partition = 'pon'
        class_per_client = num_classes

    if partition == 'pon':
        idxs = np.array(range(len(dataset_label)))
        idx_for_each_class = []
        for i in range(num_classes):
            idx_for_each_class.append(idxs[dataset_label == i])

        class_num_per_client = [class_per_client for _ in range(num_clients)]
        for i in range(num_classes):
            selected_clients = []
            for client in range(num_clients):
                if class_num_per_client[client] > 0:
                    selected_clients.append(client)
                selected_clients = selected_clients[:int(np.ceil((num_clients / num_classes) * class_per_client))]

            num_all_samples = len(idx_for_each_class[i])
            num_selected_clients = len(selected_clients)
            num_per = num_all_samples / num_selected_clients
            if balance:
                num_samples = [int(num_per) for _ in range(num_selected_clients - 1)]
            else:
                num_samples = np.random.randint(max(num_per / 10, least_samples / num_classes), num_per,
                                                num_selected_clients - 1).tolist()
            num_samples.append(num_all_samples - sum(num_samples))

            idx = 0
            for client, num_sample in zip(selected_clients, num_samples):
                if client not in dataidx_map.keys():
                    dataidx_map[client] = idx_for_each_class[i][idx:idx + num_sample]
                else:
                    dataidx_map[client] = np.append(dataidx_map[client], idx_for_each_class[i][idx:idx + num_sample],
                                                    axis=0)
                idx += num_sample
                class_num_per_client[client] -= 1

    elif partition == "dir":
        # https://github.com/IBM/probabilistic-federated-neural-matching/blob/master/experiment.py
        min_size = 0
        K = num_classes
        N = len(dataset_label)

        while min_size < least_samples:
            idx_batch = [[] for _ in range(num_clients)]
            for k in range(K):
                idx_k = np.where(dataset_label == k)[0]
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
                proportions = np.array([p * (len(idx_j) < N / num_clients) for p, idx_j in zip(proportions, idx_batch)])
                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
                min_size = min([len(idx_j) for idx_j in idx_batch])

        for j in range(num_clients):
            dataidx_map[j] = idx_batch[j]
    else:
        raise NotImplementedError

    # assign data
    for client in range(num_clients):
        idxs = dataidx_map[client]
        X[client] = dataset_content[idxs]
        y[client] = dataset_label[idxs]

        for i in np.unique(y[client]):
            statistic[client].append((int(i), int(sum(y[client] == i))))

    del data
    # gc.collect()

    for client in range(num_clients):
        print(f"Client {client}\t Size of data: {len(X[client])}\t Labels: ", np.unique(y[client]))
        print(f"\t\t Samples of labels: ", [i for i in statistic[client]])
        print("-" * 50)

    return X, y, statistic



def noniid(args,dataset, num_users):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """

    n_class =args.bingtai
    num_shards, num_imgs = num_users * n_class, int(len(dataset) / (num_users * n_class))
    idx_shard = [i for i in range(num_shards)]
    train_dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    test_dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    idxs = np.arange(len(dataset))
    labels = np.array([], dtype="int64")
    for d in dataset.datasets:
        labels = np.append(labels, np.array(d.targets, dtype='int64'))
    # labels = dataset.train_labels.numpy()
    # sort labels
    idxs_labels = np.vstack((idxs, labels))

    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # divide and assign
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, n_class, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            data = np.concatenate((train_dict_users[i], idxs[rand * num_imgs:(rand + 1) * num_imgs]), axis=0)
            train_dict_users[i], test_dict_users[i] = train_test_split(data, train_size=0.8, shuffle=True)
            # train_dict_users[i], test_dict_users[i] = data[:int(0.8*len(data))], data[int(0.8*len(data)):]
    return train_dict_users, test_dict_users



def bingtai_mnist(dataset,num_clients, num_classes_per_client, num_samples_per_class):
    # 加载MNIST数据集
    # train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True)
    # 创建一个包含所有数据索引的列表
    all_indices = list(range(len(dataset)))
    # 随机打乱索引列表
    random.shuffle(all_indices)
    dict_users = {}  # 存储划分结果的字典
    # 划分数据集
    for i in range(num_clients):
        selected_indices = []  # 当前客户端的数据索引
        # 从每个类别中选择两类
        selected_classes = random.sample(range(10), num_classes_per_client)
        # 选择每个类别的数据样本
        for class_label in selected_classes:
            indices = [index for index in all_indices if dataset.targets[index] == class_label]
            selected_indices.extend(random.sample(indices, num_samples_per_class))
        dict_users[i] = selected_indices
    draw_data_distribution(dict_users, dataset, 10)
    return dict_users


def build_noniid_agnews(dataset, num_users, alpha):
    print("DDDD1")
    train_labels = np.array([], dtype="int64")

    # 提取AG_NEWS数据集的标签
    for (text, label) in dataset:
        train_labels = np.append(train_labels, label)
    print(len(train_labels))

    n_classes = np.max(train_labels) + 1

    label_distribution = np.random.dirichlet([alpha] * num_users, n_classes)
    # (K, N)的类别标签分布矩阵X，记录每个client占有每个类别的多少

    class_idxs = [np.argwhere(train_labels == y).flatten()
                  for y in range(n_classes)]
    # 记录每个K类别对应的样本下标

    client_idxs = [[] for _ in range(num_users)]
    # 记录N个client分别对应样本集合的索引
    for c, fracs in zip(class_idxs, label_distribution):
        # np.split按照比例将类别为k的样本划分为了N个子集
        # for i, idxs 为遍历第i个client对应样本集合的索引
        for i, idxs in enumerate(np.split(c, (np.cumsum(fracs)[:-1] * len(c)).astype(int))):

            client_idxs[i] += [idxs]

    client_idxs = [np.concatenate(idxs) for idxs in client_idxs]
    #
    train_dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    test_dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}

    for i in range(len(client_idxs)):
        data = client_idxs[i]
        train_dict_users[i], test_dict_users[i] = train_test_split(data, train_size=0.8, shuffle=True)

    # draw_data_distribution(train_dict_users, dataset, n_classes)
    # draw_data_distribution(test_dict_users, dataset, n_classes)
    return train_dict_users, test_dict_users


def build_noniid(dataset, num_users, alpha):
    print("DDDD1")
    train_labels = np.array([], dtype="int64")
    for d in dataset.datasets:
        train_labels = np.append(train_labels, np.array(d.targets, dtype='int64'))
    # train_labels = np.array(dataset.targets)
    n_classes = np.max(train_labels) + 1
    label_distribution = np.random.dirichlet([alpha] * num_users, n_classes)
    # (K, N)的类别标签分布矩阵X，记录每个client占有每个类别的多少

    class_idxs = [np.argwhere(train_labels == y).flatten()
                  for y in range(n_classes)]
    # 记录每个K类别对应的样本下标

    client_idxs = [[] for _ in range(num_users)]
    # 记录N个client分别对应样本集合的索引
    for c, fracs in zip(class_idxs, label_distribution):
        # np.split按照比例将类别为k的样本划分为了N个子集
        # for i, idxs 为遍历第i个client对应样本集合的索引
        for i, idxs in enumerate(np.split(c, (np.cumsum(fracs)[:-1] * len(c)).astype(int))):
            client_idxs[i] += [idxs]

    client_idxs = [np.concatenate(idxs) for idxs in client_idxs]
    #
    train_dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    test_dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}

    for i in range(len(client_idxs)):
        data = client_idxs[i]
        train_dict_users[i], test_dict_users[i] = train_test_split(data, train_size=0.8, shuffle=True)

    # draw_data_distribution(train_dict_users, dataset, n_classes)
    # draw_data_distribution(test_dict_users, dataset, n_classes)
    return train_dict_users, test_dict_users


def draw_data_distribution(dict_users, dataset, num_class):
    import matplotlib.pyplot as plt
    targets = dataset.targets

    # plt.figure(figsize=(20, 3))
    plt.hist([np.array(targets)[idc] for idc in dict_users.values()], stacked=True,
             bins=np.arange(min(targets) - 0.5, max(targets) + 1.5, 1),
             label=["C{}".format(i) for i in range(len(dict_users))], rwidth=0.5)
    plt.xticks(np.arange(num_class), rotation=70)
    plt.legend(loc=(0.95, -0.1))
    plt.savefig("2.jpg")
    plt.show()


if __name__ == '__main__':

    trans_fashion_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

    dataset_train = datasets.FashionMNIST('../data/fashion-mnist', train=True, download=True,
                                          transform=trans_fashion_mnist)

    train_dict_users, test_dict_users = noniid(dataset_train, 10)
    draw_data_distribution(train_dict_users, dataset_train, 10)
    draw_data_distribution(test_dict_users, dataset_train, 10)
    # num = 100
    # d = mnist_iid(dataset_train, num)
    # path = '../data/fashion_iid_100clients.dat'
    # file = open(path, 'w')
    # for idx in range(num):
    #     for i in d[idx]:
    #         file.write(str(i))
    #         file.write(',')
    #     file.write('\n')
    # file.close()
    # trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    # dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True, transform=trans_mnist)
    # print(fashion_iid(dataset_train, 1000)[0])


