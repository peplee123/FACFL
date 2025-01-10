#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import matplotlib
import matplotlib.pyplot as plt
import copy
import numpy as np
from torchvision import datasets, transforms
import torch
import os
import random
from utils.sampling import noniid, build_noniid,build_noniid_agnews, separate_data
from utils.options import args_parser
from utils.dataset import CustomAGNewsDataset
from models.Update import DatasetSplit, LocalUpdate
from models.Nets import LeNet5Cifar,LeNet5fm,LeNet5,LeNet5Fmnist,resnet18,MLP, CNNMnist, CNNCifar, CNNFemnist, fastText, CNNTinyImage, CNNCifar100,ResNet9
from models.Fed import FedAvg,FimBA,NewFedBa,diff
from models.test import test_img
from utils.dataset import FEMNIST, ShakeSpeare, ImageFolder_custom, CustomImageDataset
from torch.utils.data import ConcatDataset, Dataset
import torch
import seaborn as sns
from scipy.stats import entropy
from torchvision import transforms, datasets
from torch.utils.data import ConcatDataset
from opacus.grad_sample import GradSampleModule

if __name__ == '__main__':
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')




    if args.dataset == 'svhn':

        trans_svhn = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        # 下载和加载训练集和测试集
        dataset_train = datasets.SVHN(root='../Federated-Learning-BA/data/dataset/svhn/', split='train', download=True, transform=trans_svhn)
        dataset_test = datasets.SVHN(root='../Federated-Learning-BA/data/dataset/svhn/', split='test', download=True, transform=trans_svhn)

        if args.iid:
            dict_users = svhn_iid(dataset_train, args.num_users)  # 注意：你需要定义一个svhn_iid函数或者复用mnist_iid
        else:
            # 合并训练集和测试集
            dataset_train = ConcatDataset([dataset_train, dataset_test])
            if args.type == 'dir':
                print("dir")
                train_dict_users, test_dict_users = build_noniid(dataset_train, args.num_users, args.dir)
                print(len(train_dict_users),print(len(test_dict_users)))
            elif args.type == 'pon':
                print("Pon")
                train_dict_users, test_dict_users = noniid(args, dataset_train, args.num_users)
            else:
                print("type is none")


    elif args.dataset == 'mnist':
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.MNIST('/data/dataset/mnist', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST('/data/dataset/mnist', train=False, download=True, transform=trans_mnist)

        if args.iid:
            dict_users = mnist_iid(dataset_train, args.num_users)
        else:
            dataset_train = ConcatDataset([dataset_train, dataset_test])
            if args.type == 'dir':
                print("dir")
                train_dict_users, test_dict_users = build_noniid(dataset_train, args.num_users, args.dir)
            elif args.type == 'pon':
                print("Pon")
                train_dict_users, test_dict_users = noniid(args, dataset_train, args.num_users)
            else:
                print("type is none")
    elif args.dataset == 'cifar10':
        #trans_cifar = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        trans_cifar_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        trans_cifar_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        dataset_train = datasets.CIFAR10('/data/dataset/cifar10', train=True, download=True, transform=trans_cifar_train)
        dataset_test = datasets.CIFAR10('/data/dataset/cifar10', train=False, download=True, transform=trans_cifar_test)
        if args.iid:
            dict_users = cifar_iid(dataset_train, args.num_users)
        else:
            dataset_train = ConcatDataset([dataset_train, dataset_test])
            if args.type == 'dir':
                print("dir")
                train_dict_users, test_dict_users = build_noniid(dataset_train, args.num_users, args.dir)
            elif args.type == 'pon':
                print("Pon")
                train_dict_users, test_dict_users = noniid(args, dataset_train, args.num_users)
            else:
                print("type is none")
        # from scipy.stats import entropy
        # import numpy as np
        # import matplotlib.pyplot as plt
        # import seaborn as sns
        # from collections import Counter
        #
        #
        # def calculate_kl_divergence(train_dict_users, dataset, num_classes=10, epsilon=1e-10):
        #     # 获取每个客户端的标签分布
        #     client_distributions = []
        #     for client_data in train_dict_users.values():
        #         # 提取每个客户端的数据索引对应的标签
        #         labels = [dataset[idx][1] for idx in client_data]  # 假设 dataset[idx][1] 是标签
        #         label_counts = Counter(labels)
        #
        #         # 创建一个长度为 num_classes 的分布数组
        #         distribution = np.zeros(num_classes)
        #         for label, count in label_counts.items():
        #             distribution[label] = count
        #         distribution = distribution / distribution.sum()  # 归一化
        #
        #         # 添加一个小的平滑项，避免零概率
        #         distribution = distribution + epsilon
        #         distribution = distribution / distribution.sum()  # 再次归一化
        #
        #         client_distributions.append(distribution)
        #
        #     print("client_distributions", client_distributions)  # 打印以检查分布
        #
        #     # 计算客户端之间的 KL 散度
        #     num_clients = len(train_dict_users)
        #     kl_divergences = np.zeros((num_clients, num_clients))
        #     for i in range(num_clients):
        #         for j in range(num_clients):
        #             if i != j:
        #                 kl_ij = entropy(client_distributions[i], client_distributions[j])
        #                 kl_ji = entropy(client_distributions[j], client_distributions[i])
        #                 kl_divergences[i, j] = (kl_ij + kl_ji) / 2  # 取平均值以保证对称性
        #
        #     return kl_divergences
        #
        #
        # # # 生成热力图
        # kl_matrix = calculate_kl_divergence(train_dict_users, dataset_train, num_classes=10)
        # plt.figure(figsize=(10, 8))
        # sns.heatmap(kl_matrix, cmap="magma", square=True)
        # plt.xlabel("client id")
        # plt.ylabel("client id")
        # # plt.title("Heatmap of KL Divergence between Clients")
        # plt.show()
        # plt.savefig("kl.jpg")
        #
        # from scipy.spatial.distance import jensenshannon
        # import numpy as np
        # import matplotlib.pyplot as plt
        # import seaborn as sns
        # from collections import Counter
        #
        #
        # def calculate_js_divergence(train_dict_users, dataset, num_classes=10):
        #     # 获取每个客户端的标签分布
        #     client_distributions = []
        #     for client_data in train_dict_users.values():
        #         # 提取每个客户端的数据索引对应的标签
        #         labels = [dataset[idx][1] for idx in client_data]  # 假设 dataset[idx][1] 是标签
        #         label_counts = Counter(labels)
        #
        #         # 创建一个长度为 num_classes 的分布数组
        #         distribution = np.zeros(num_classes)
        #         for label, count in label_counts.items():
        #             distribution[label] = count
        #         distribution = distribution / distribution.sum()  # 归一化
        #         client_distributions.append(distribution)
        #
        #     print("client_distributions", client_distributions)  # 打印以检查分布
        #
        #     # 计算客户端之间的 Jensen-Shannon 散度
        #     num_clients = len(train_dict_users)
        #     js_divergences = np.zeros((num_clients, num_clients))
        #     for i in range(num_clients):
        #         for j in range(num_clients):
        #             if i != j:
        #                 js_divergences[i, j] = jensenshannon(client_distributions[i], client_distributions[j])
        #
        #     return js_divergences
        #
        #
        # # 生成热力图
        # js_matrix = calculate_js_divergence(train_dict_users, dataset_train, num_classes=10)
        # plt.figure(figsize=(10, 8))
        # sns.heatmap(js_matrix, cmap="magma", square=True, cbar_kws={"label": "JS Divergence"})
        # plt.xlabel("client id")
        # plt.ylabel("client id")
        # plt.title("Heatmap of JS Divergence between Clients")
        # plt.show()
        # plt.savefig("js.jpg")

    elif args.dataset == 'cifar100':
        #trans_cifar = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        trans_cifar_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        trans_cifar_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        dataset_train = datasets.CIFAR100('/data/dataset/cifar100', train=True, download=True, transform=trans_cifar_train)
        dataset_test = datasets.CIFAR100('/data/dataset/cifar100', train=False, download=True, transform=trans_cifar_test)
        if args.iid:
            dict_users = cifar_iid(dataset_train, args.num_users)
        else:
            dataset_train = ConcatDataset([dataset_train, dataset_test])
            if args.type == 'dir':
                print("dir")
                train_dict_users, test_dict_users = build_noniid(dataset_train, args.num_users, args.dir)
            elif args.type == 'pon':
                print("Pon")
                train_dict_users, test_dict_users = noniid(args, dataset_train, args.num_users)
            else:
                print("type is none")
    elif args.dataset == 'fashion-mnist':
        trans_fashion_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        dataset_train = datasets.FashionMNIST('./data/fashion-mnist', train=True, download=True,
                                              transform=trans_fashion_mnist)
        dataset_test = datasets.FashionMNIST('./data/fashion-mnist', train=False, download=True,
                                              transform=trans_fashion_mnist)
        if args.iid:
            dict_users = mnist_iid(dataset_train, args.num_users)
        else:

            dataset_train = ConcatDataset([dataset_train, dataset_test])
            if args.type == 'dir':
                print("dir")
                train_dict_users, test_dict_users = build_noniid(dataset_train, args.num_users, args.dir)
            elif args.type == 'pon':
                print("Pon")
                train_dict_users, test_dict_users = noniid(args, dataset_train, args.num_users)
            else:
                print("type is none")
    elif args.dataset == 'femnist':
        dataset_train = FEMNIST(train=True)
        dataset_test = FEMNIST(train=False)
        dict_users = dataset_train.get_client_dic()
        args.num_users = len(dict_users)
        if args.iid:
            exit('Error: femnist dataset is naturally non-iid')
        else:
            dataset_train = ConcatDataset([dataset_train, dataset_test])
            if args.type == 'dir':
                print("dir")
                train_dict_users, test_dict_users = build_noniid(dataset_train, args.num_users, args.dir)
            elif args.type == 'pon':
                print("Pon")
                train_dict_users, test_dict_users = noniid(args, dataset_train, args.num_users)
            else:
                print("type is none")
    elif args.dataset == "tiny-image":
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        trainset = ImageFolder_custom(root='/data/dataset/tiny-imagenet-200/train/', transform=transform)
        testset = ImageFolder_custom(root='/data/dataset/tiny-imagenet-200/val/', transform=transform)
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=len(trainset), shuffle=False)
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=len(testset), shuffle=False)

        for _, train_data in enumerate(trainloader, 0):
            trainset.data, trainset.targets = train_data
        for _, test_data in enumerate(testloader, 0):
            testset.data, testset.targets = test_data

        dataset_image = []
        dataset_label = []

        dataset_image.extend(trainset.data.cpu().detach().numpy())
        dataset_image.extend(testset.data.cpu().detach().numpy())
        dataset_label.extend(trainset.targets.cpu().detach().numpy())
        dataset_label.extend(testset.targets.cpu().detach().numpy())
        dataset_image = np.array(dataset_image)
        dataset_label = np.array(dataset_label)

        if args.iid:
            exit('Error: not support')
        else:
            dataset_train = ConcatDataset([CustomImageDataset(dataset_image, dataset_label)])
            if args.type == 'dir':
                print("dir")
                train_dict_users, test_dict_users = build_noniid(dataset_train, args.num_users, args.dir)
            elif args.type == 'pon':
                print("Pon")
                train_dict_users, test_dict_users = noniid(args, dataset_train, args.num_users)
            else:
                print("type is none")


    elif args.dataset == "P":
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        trainset = ImageFolder_custom(root='/data/dataset/tiny-imagenet-200/train/', transform=transform)
        testset = ImageFolder_custom(root='/data/dataset/tiny-imagenet-200/val/', transform=transform)
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=len(trainset), shuffle=False)
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=len(testset), shuffle=False)

        for _, train_data in enumerate(trainloader, 0):
            trainset.data, trainset.targets = train_data
        for _, test_data in enumerate(testloader, 0):
            testset.data, testset.targets = test_data

        dataset_image = []
        dataset_label = []

        dataset_image.extend(trainset.data.cpu().detach().numpy())
        dataset_image.extend(testset.data.cpu().detach().numpy())
        dataset_label.extend(trainset.targets.cpu().detach().numpy())
        dataset_label.extend(testset.targets.cpu().detach().numpy())
        dataset_image = np.array(dataset_image)
        dataset_label = np.array(dataset_label)

        if args.iid:
            exit('Error: not support')
        else:
            dataset_train = ConcatDataset([CustomImageDataset(dataset_image, dataset_label)])
            if args.type == 'dir':
                print("dir")
                train_dict_users, test_dict_users = build_noniid(dataset_train, args.num_users, args.dir)
            elif args.type == 'pon':
                print("Pon")
                train_dict_users, test_dict_users = noniid(args, dataset_train, args.num_users)
            else:
                print("type is none")
    # img_size = dataset_train[0][0].shape

    # build model
    if args.model == 'cnn' and args.dataset == 'cifar10':
        net_glob = LeNet5(num_classes=10).to(args.device)
        print("model is lenet")
    elif args.model == 'cnn' and args.dataset == 'cifar100':
        net_glob =ResNet9(in_channels=3,num_classes=100).to(args.device)
        print("model is ResNet")
    elif args.model == 'cnn' and (args.dataset == 'mnist' or args.dataset == 'fashion-mnist'):
        net_glob = LeNet5fm(num_classes=10).to(args.device)
        print("model is lenet")
    elif args.dataset == 'femnist' and args.model == 'cnn':
        net_glob = CNNFemnist(args=args).to(args.device)
    elif args.dataset == 'shakespeare' and args.model == 'lstm':
        net_glob = CharLSTM().to(args.device)
    elif args.dataset == 'agnews':
        net_glob = fastText(hidden_dim=32, vocab_size=98635, num_classes=4).to(args.device)
    elif args.dataset == 'tiny-image':
        net_glob = resnet18(num_classes=200).to(args.device)
    elif args.model == 'cnn' and args.dataset == 'svhn':
        net_glob = LeNet5(num_classes=10).to(args.device)
    else:
        exit('Error: unrecognized model')
    net_glob.train()
    w_glob = copy.deepcopy(net_glob)

    # training
    #定义一个客户端的编号表
    num_client = []
    #定义一个聚类表字典
    cluster_count = args.num_users
    cluster_dict = {k: [] for k in range(cluster_count+1)}
    cluster_dict[0] = range(args.num_users)
    cluster_model_list = [copy.deepcopy(net_glob) for _ in range(cluster_count+1)]
    client_model_list = [copy.deepcopy(net_glob) for _ in range(args.num_users)]
    acc_test = []
    acc_client = []
    loss_train = []
    total_comm_cost_list = []
    learning_rate = [args.lr for i in range(args.num_users)]
    total_comm_cost = 0.0  # 初始化总通讯量

    user_local_dict = {}

    for i in range(args.num_users):
        user_local_dict[i] = LocalUpdate(args=args, dataset=dataset_train, idxs=train_dict_users[i],
                                         test_idxs=test_dict_users[i])
    # for i in range(args.num_users):
    #     if args.serial:
    #         print("DPS")
    #         user_local_dict[i] = LocalUpdateDPSerial(args=args, dataset=dataset_train, idxs=train_dict_users[i], test_idxs=test_dict_users[i])
    #     else:
    #         user_local_dict[i] = LocalUpdateDP(args=args, dataset=dataset_train, idxs=train_dict_users[i], test_idxs=test_dict_users[i])


    for iter in range(args.epochs):
        allclient_distributed = []
        w_locals, loss_locals,  fim_local ,modeldiff_local = [], [], [],[]
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        print('选取的客户端编号', idxs_users)
        acc_total = 0
        for idx in idxs_users:
            args.lr = learning_rate[idx]
            local = user_local_dict[idx]
            # local = LocalUpdate(args=args, dataset=dataset_train, idxs=train_dict_users[idx], test_idxs=test_dict_users[idx])
            w, loss, curLR,everyclient_distributed, acc ,fim ,param_diff,total_comm_cost= local.train(net=copy.deepcopy(client_model_list[idx]).to(args.device),total_comm_cost=total_comm_cost)
            acc_total += acc
            learning_rate[idx] = curLR
            w_locals.append(copy.deepcopy(w))
            loss_locals.append(copy.deepcopy(loss))
            fim_local.append(copy.deepcopy(fim))
            modeldiff_local.append(copy.deepcopy(param_diff))
            allclient_distributed.append(everyclient_distributed)
        acc = acc_total/len(idxs_users)
        print(f"avg acc {acc}")
        acc_client.append(acc)
        tensor_list = [item[0] for item in allclient_distributed]
        if args.juleiyiju == 'fisher':
            w_global_dict, index_dict = FimBA(w_locals, fim_local, tensor_list,args.maxcluster,args)
        elif args.juleiyiju == 'soft':
            w_global_dict, index_dict = NewFedBa(w_locals, tensor_list, args.maxcluster)
        elif args.juleiyiju == "model":
            w_global_dict, index_dict = diff(w_locals, modeldiff_local, args.maxcluster)

        cluster_acc_total = 0
        for k, idx_list in index_dict.items():
            for idx in idx_list:
                c_id = idxs_users[idx]
                # w_global_dict[k] =interpolate_models(w_global_dict[k],w_glob,0.5)
                # w_locals[idx] = copy.deepcopy(w_global_dict[k])
                client_model_list[c_id].load_state_dict(w_global_dict[k])
            cluster_model_list[k].load_state_dict(w_global_dict[k])
            cluster_dict[k] = idx_list

            dataset_test_idx = np.concatenate([test_dict_users[idxs_users[idx]] for idx in idx_list])

            # print accuracy
            acc_t, loss_t = test_img(cluster_model_list[k], DatasetSplit(dataset_train, dataset_test_idx), args)
            cluster_acc_total += acc_t
            # print("Round {:3d},cluster {} Testing accuracy: {:.2f}".format(iter, k, acc_t))
        acc_Cluster = cluster_acc_total / len(index_dict)
        acc_test.append(acc_Cluster)
        print(f"Round {iter:3d}, cluster avg acc {acc_Cluster}")
        cost_in_mb = total_comm_cost / 1024 / 1024  # Convert to MB
        cost_in_mb_per_user = cost_in_mb / 20  # Normalize by a factor of 20
        total_comm_cost_list.append(cost_in_mb_per_user)
        print(total_comm_cost_list)
        # if acc >= 80 :
        #     print(f"目标精度 {80}% 达成，停止代码运行！")
        #     print(f"总通讯开销: {total_comm_cost:.2f} MB")
        #     exit()  # 直接退出整个程序

        # copy weight to net_glob
        # net_glob.load_state_dict(w_glob)
        # net_glob.load_state_dict(w_glob.state_dict())
        loss_avg = sum(loss_locals) / len(loss_locals)
        print('Round {:3d}, Average loss {:.3f}'.format(iter, loss_avg))
        loss_train.append(loss_avg)

        rootpath = './12.25最新实验'
        if not os.path.exists(rootpath):
            os.makedirs(rootpath)
        accfile = open(rootpath + '/acc_cluster_avg_file_fed_{}_{}_{}_iid{}_{}_{}_{}_{}_{}_{}.dat'.
                       format(args.dataset, args.model, args.epochs, args.iid,args.lr,args.local_bs,args.beizhu,args.dp_mechanism,args.juleiyiju,args.juhe), "w")
        accfile1 = open(rootpath + '/loss_file_fed_{}_{}_{}_iid{}_{}_{}_{}_{}_{}_{}.dat'.
                       format(args.dataset, args.model, args.epochs, args.iid,args.lr,args.local_bs,args.beizhu,args.dp_mechanism,args.juleiyiju,args.juhe), "w")
        accfile2 = open(rootpath + '/acc_client_avg_file_fed_{}_{}_{}_iid{}_{}_{}_{}_{}_{}_{}.dat'.
                        format(args.dataset, args.model, args.epochs, args.iid, args.lr, args.local_bs,args.beizhu,args.dp_mechanism,args.juleiyiju,args.juhe), "w")
        accfile3 = open(rootpath + '/communication_client_avg_file_fed_{}_{}_{}_iid{}_{}_{}_{}_{}_{}_{}.dat'.
                        format(args.dataset, args.model, args.epochs, args.iid, args.lr, args.local_bs,args.beizhu,args.dp_mechanism,args.juleiyiju,args.juhe), "w")
        for ac in acc_test:
            sac = str(ac)
            accfile.write(sac)
            accfile.write('\n')
        accfile.close()
        for ac in loss_train:
            sac = str(ac)
            accfile1.write(sac)
            accfile1.write('\n')
        accfile1.close()
        for ac in acc_client:
            sac = str(ac)
            accfile2.write(sac)
            accfile2.write('\n')
        accfile2.close()
        for ac in total_comm_cost_list:
            sac = str(ac)
            accfile3.write(sac)
            accfile3.write('\n')
        accfile3.close()

    # plot loss curve
    plt.figure()
    plt.plot(range(len(acc_client)), acc_client)
    plt.ylabel('test accuracy')
    plt.savefig(rootpath + '/fed_{}_{}_{}_C{}_iid{}_{}_{}_acc_client_{}_{}_{}.png'.format(args.dataset, args.model, args.epochs, args.frac, args.iid,args.lr,args.local_bs,args.beizhu,args.juleiyiju,args.juhe))



