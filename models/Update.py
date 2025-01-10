#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import torch
from torch import nn, autograd
from torch.utils.data import DataLoader, Dataset
from models.test import test_img
import torch.nn.functional as F
import copy
import numpy as np
from utils.noise import add_noise
import matplotlib.pyplot as plt
import torch.optim as optim
import torch
from torch.autograd import grad



def distillation_loss(outputs, teacher_outputs, temperature):
    soft_labels = nn.functional.softmax(teacher_outputs / temperature, dim=1)
    log_probs = nn.functional.log_softmax(outputs, dim=1)
    loss = nn.functional.kl_div(log_probs, soft_labels, reduction='batchmean') * temperature ** 2
    return loss

def bhattacharyya_distance(vector1, vector2):
    # Avoid division by zero
    epsilon = 1e-10
    vector1 = torch.mean(vector1, dim=0).cpu().detach().numpy()
    vector2 = torch.mean(vector2, dim=0).cpu().detach().numpy()
    vector1 = np.clip(vector1, epsilon, 1.0 - epsilon)
    vector2 = np.clip(vector2, epsilon, 1.0 - epsilon)
    BC = np.sum(np.sqrt(vector1 * vector2))
    return -np.log(BC)

class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


def copy_layers(old_net, new_net, n):
    """
    将 old_net 模型的前 n 层参数复制给 new_net 模型。

    参数：
    old_net (torch.nn.Module): 要复制参数的原始模型。
    new_net (torch.nn.Module): 要接受参数复制的新模型。
    n (int): 要复制的层数。

    注意：需要确保 old_net 和 new_net 模型具有相同的层结构，以便复制参数。
    """
    if n < 1:
        raise ValueError("n 必须大于等于 1")

    old_params = list(old_net.parameters())
    new_params = list(new_net.parameters())

    if len(old_params) < n or len(new_params) < n:
        raise ValueError("模型的参数数量少于 n")

    for i in range(n):
        new_params[i].data = copy.deepcopy(old_params[i].data)

def frobenius_norm_diff(H_g, H_l):
    """ 计算两个 Hessian 矩阵的 Frobenius 范数差 """
    diff = H_g - H_l
    frobenius_norm = torch.sqrt(torch.sum(diff ** 2))
    return frobenius_norm

def compute_jacobian(input_tensor, model, target_layer):
    model.eval()
    input_tensor = input_tensor.requires_grad_(True)
    output = model(input_tensor)
    jacobian_matrix = []

    for i in range(output.size(1)):  # 对输出的每个维度进行
        grad_output = torch.zeros_like(output)
        grad_output[:, i] = 1
        jacobian_i = autograd.grad(outputs=output, inputs=target_layer, grad_outputs=grad_output,
                                   create_graph=True)[0]
        jacobian_matrix.append(jacobian_i)

    jacobian_matrix = torch.stack(jacobian_matrix, dim=1)
    return jacobian_matrix
# 获取模型的基础层（例如前n层）
def get_base_layers(model, n):
    # 获取模型参数的列表
    params = list(model.parameters())
    # 取出前n层的参数
    base_params = params[:n]  # 基础层通常是前n层
    return base_params


def calculate_base_layer_size(model, n):
    base_params = get_base_layers(model, n)
    total_params = 0
    for param in base_params:
        total_params += param.numel()  # 获取该层参数的元素个数
    total_size_in_bytes = total_params * 4  # 每个参数是32位浮动数，占用4字节
    # total_size_in_mb = total_size_in_bytes / (1024 ** 2)  # 转换为MB
    return total_size_in_bytes




def calculate_fisher_info_size(fim_vector):
    fim_vector = fim_vector * 1e9  # 放大数值
    fim_vector = fim_vector.to(torch.float64)  # 转换为 float64 类型
    total_fisher_elements = fim_vector.numel()  # Fisher 信息矩阵的元素数量
    total_size_in_bytes = total_fisher_elements * 8  # 每个元素是64位浮动数，占8字节
    # total_size_in_mb = total_size_in_bytes / (1024 ** 2)  # 转换为 MB
    return total_size_in_bytes

def calculate_comm_cost(model, fim_vector):
    model_size = calculate_model_size(model)
    fisher_info_size = calculate_fisher_info_size(fim_vector)
    total_comm_cost = model_size + fisher_info_size
    return total_comm_cost


class LocalUpdate(object):
    def __init__(self, args, dataset=None, idxs=None, test_idxs=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)
        self.ldr_test = DataLoader(DatasetSplit(dataset, test_idxs), batch_size=self.args.local_bs, shuffle=False)
        self.last_net = None


    def train(self, net,total_comm_cost):
        print("train")
        if self.last_net:
            n=len( list(net.parameters()))
            print('========',n)
            copy_layers(net, self.last_net, n-self.args.layers)
            net = copy.deepcopy(self.last_net)
        global_w = copy.deepcopy(net)
        net.train()
        global_w.eval()
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum,weight_decay=1e-3)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=self.args.lr_decay)
        specific_layers = [list(net.parameters())[-2], list(net.parameters())[-1]]
        jacobian_optimizer = torch.optim.SGD(specific_layers, lr=self.args.lr, momentum=self.args.momentum, weight_decay=1e-3)
        epoch_loss = []
        everyclient_distributed =[]
        total_local_probs = 0
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                net.zero_grad()
                log_probs = net(images)
                global_probs = global_w(images)
                local_probs = F.softmax(log_probs, dim=1)
                global_probs = F.softmax(global_probs, dim=1)
                # # 获取拼接处的层
                # layer_g = list(net.parameters())[-2]  # 基础层最后一层
                # layer_l = list(net.parameters())[-1]  # 个性化层第一层
                # jacobian_g = compute_jacobian(images, net, layer_g)
                # jacobian_l = compute_jacobian(images, net, layer_l)

                # # 提取张量并计算范数
                # norm1 = torch.norm(jacobian_g, p='fro')
                # norm2 = torch.norm(jacobian_l, p='fro')
                #
                # # 计算范数的差异
                # frobenius_diff = torch.abs((norm1 - norm2) / (norm1 + norm2 + 1e-8))
                #
                proximal_term = bhattacharyya_distance(local_probs,global_probs)
                # loss1= frobenius_diff
                loss= self.loss_func(log_probs, labels)+self.args.hy*proximal_term

                # loss1.backward()
                loss.backward()
                # jacobian_optimizer.step()
                optimizer.step()
                scheduler.step()
                total_local_probs += local_probs.sum(dim=0)

                batch_loss.append(loss.item())
                # print("loss: ", loss)
            epoch_loss.append(sum(batch_loss) / len(batch_loss))

        # 遍历 net 和 global_w 的参数，计算差异
        param_diff = 0.0
        for p_net, p_global in zip(net.parameters(), global_w.parameters()):
            param_diff += torch.norm(p_net - p_global, p='fro').item()

        # -------------计算fisher-----------
        # 设置模型为评估模式
        net.eval()

        # 用于存储每个参数的梯度平方（即 Fisher 信息矩阵对角线元素）
        fim_diagonal = []

        # 遍历训练数据
        for i, (x, y) in enumerate(self.ldr_train):
            # 前向传播
            x = x.to(self.args.device)
            y = y.to(self.args.device)
            outputs = net(x)

            # 负对数似然损失
            nll = -torch.nn.functional.log_softmax(outputs, dim=1)[range(len(y)), y].mean()

            # 计算损失相对于模型参数的梯度
            grads = grad(nll, net.parameters(), create_graph=True)

            # 遍历每个梯度，计算 Fisher 信息矩阵对角线元素
            for g in grads:
                # 检查梯度是否为空
                if g is not None:
                    fim_diagonal.append(torch.sum(g ** 2).item())  # 使用 .item() 转换为数值
                else:
                    fim_diagonal.append(0)

        # 将 Fisher 信息矩阵的对角线元素保存为向量
        fim_vector = torch.tensor(fim_diagonal)
        # 计算通讯量
        base_layer_size = calculate_base_layer_size(net, 8)
        fisher_info_size = calculate_fisher_info_size(fim_vector)
        comm_cost = base_layer_size + fisher_info_size
        total_comm_cost += comm_cost  # 累加通讯量

        print(f"基础层的大小: {base_layer_size:.2f} B")
        print(f"Fisher 信息矩阵大小: {fisher_info_size:.2f} B")
        print(f"本轮通讯开销: {comm_cost:.2f} B")
        print(f"累计通讯开销: {total_comm_cost:.2f} B")

        # 计算平均软预测
        sum_ = sum(total_local_probs)
        total_local_probs = torch.tensor([p/sum_ for p in total_local_probs])
        everyclient_distributed.append(total_local_probs)
        # 计算平均软预测

        # accuracyafter, test_loss = test_img(DPnet, self.ldr_test.dataset, self.args)
        accuracybefor, test_loss1 = test_img(global_w, self.ldr_test.dataset, self.args)
        # 如果达到目标精度，则停止代码

        self.last_net = copy.deepcopy(net)
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss), scheduler.get_last_lr()[0],everyclient_distributed, accuracybefor,fim_vector,param_diff, total_comm_cost

