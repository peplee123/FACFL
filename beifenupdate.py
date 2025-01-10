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
from collections import defaultdict


class SAM:
    '''
    Sharpness-Aware Minimization Optimizer
    '''

    def __init__(self, optimizer, model, rho=0.5):
        self.optimizer = optimizer
        self.model = model
        self.rho = rho
        self.state = defaultdict(dict)

    @torch.no_grad()
    def ascent_step(self):
        grads = []
        layers_to_optimize = list(self.model.children())[-2:]  # 只取倒数第二层和倒数第三层
        for layer in layers_to_optimize:
            for n, p in layer.named_parameters():
                if p.grad is None:
                    continue
                grads.append(torch.norm(p.grad, p=2))
        grad_norm = torch.norm(torch.stack(grads), p=2) + 1.e-16
        for layer in layers_to_optimize:
            for n, p in layer.named_parameters():
                if p.grad is None:
                    continue
                eps = self.state[p].get("eps")
                if eps is None:
                    eps = torch.clone(p).detach()
                    self.state[p]["eps"] = eps
                eps[...] = p.grad[...]
                eps.mul_(self.rho / grad_norm)
                p.add_(eps)
        self.optimizer.zero_grad()

    @torch.no_grad()
    def descent_step(self):
        layers_to_optimize = list(self.model.children())[-2:]  # 只取倒数第二层和倒数第三层
        for layer in layers_to_optimize:
            for n, p in layer.named_parameters():
                if p.grad is None:
                    continue
                p.sub_(self.state[p]["eps"])
        self.optimizer.step()
        self.optimizer.zero_grad()


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


def compute_frobenius_difference(hessian_g, hessian_l):
    # 计算 Frobenius 范数
    frobenius_g = torch.norm(hessian_g, p='fro')
    frobenius_l = torch.norm(hessian_l, p='fro')

    # 计算差异
    frobenius_diff = torch.abs(frobenius_g - frobenius_l)

    return frobenius_diff


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


def spectral_norm_diff(H_g, H_l):
    """ 计算两个 Hessian 矩阵的谱范数差异 """
    # 计算每个矩阵的谱范数
    spectral_norm_g = torch.linalg.norm(H_g, ord=2)
    spectral_norm_l = torch.linalg.norm(H_l, ord=2)

    # 计算差异
    spectral_norm_diff = torch.abs(spectral_norm_g - spectral_norm_l)
    return spectral_norm_diff


def cosine_similarity_jacobians(jacobian1, jacobian2):
    # 将矩阵展平成向量
    j1_flat = jacobian1.view(-1)
    j2_flat = jacobian2.view(-1)

    # 计算余弦相似性
    cos_sim = F.cosine_similarity(j1_flat, j2_flat, dim=0)
    return cos_sim.item()


class LocalUpdate(object):
    def __init__(self, args, dataset=None, idxs=None, test_idxs=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)
        self.ldr_test = DataLoader(DatasetSplit(dataset, test_idxs), batch_size=self.args.local_bs, shuffle=False)
        self.last_net = None

    # def compute_hessian(self, model, data, target, layer_g, layer_l):
    # def compute_hessian(self, loss, layer_g, layer_l):
    #     # model.eval()
    #     # output = model(data)
    #     # loss = self.loss_func(output, target)
    #
    #     grads_g = autograd.grad(loss, layer_g, create_graph=True)
    #     grads_l = autograd.grad(loss, layer_l, create_graph=True)
    #
    #     hessian_g = []
    #     hessian_l = []
    #     # 将梯度求和以获得标量输出
    #     grads_g_sum = sum([g.sum() for g in grads_g])
    #     grads_l_sum = sum([g.sum() for g in grads_l])
    #
    #     # 这里不再迭代，而是直接对标量进行操作
    #     hessian_g.append(autograd.grad(grads_g_sum, layer_g, retain_graph=True, create_graph=True))
    #     hessian_l.append(autograd.grad(grads_l_sum, layer_l, retain_graph=True, create_graph=True))
    #
    #     return hessian_g, hessian_l

    # def compute_cosine_similarity(self, grads_g, grads_l):
    #     # 计算梯度的范数
    #     grads_g_norm = grads_g.norm()
    #     grads_l_norm = grads_l.norm()
    #
    #     # 计算范数的余弦相似性
    #     cosine_similarity = F.cosine_similarity(grads_g_norm.view(-1), grads_l_norm.view(-1), dim=0)
    #
    #     return cosine_similarity.item()

    def train(self, net):
        print("train")
        if self.last_net:
            n=len( list(net.parameters()))
            print('========',n)
            copy_layers(net, self.last_net, n-self.args.layers)
            net = copy.deepcopy(self.last_net)
        global_w = copy.deepcopy(net)
        net.train()
        global_w.eval()
        # train and update
        # optimizer = torch.optim.Adam(net.parameters(),lr=self.args.lr,weight_decay=1e-3)
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum,weight_decay=1e-3)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=self.args.lr_decay)
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=self.args.lr_decay)

        # 新的优化器，仅用于优化指定的层
        specific_layers = [list(net.parameters())[-2], list(net.parameters())[-1]]
        # jacobian_optimizer = torch.optim.SGD(specific_layers, lr=self.args.lr, momentum=self.args.momentum, weight_decay=1e-3)
        # 使用 SAM 优化器
        jacobian_optimizer = SAM(optimizer, net, 0.1)
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

                # 获取拼接处的层
                layer_g = list(net.parameters())[-2]  # 基础层最后一层
                layer_l = list(net.parameters())[-1]  # 个性化层第一层
                jacobian_g = compute_jacobian(images, net, layer_g)
                jacobian_l = compute_jacobian(images, net, layer_l)

                # 提取张量并计算范数
                norm1 = torch.norm(jacobian_g, p='fro')
                norm2 = torch.norm(jacobian_l, p='fro')

                # 计算范数的差异
                frobenius_diff = torch.abs((norm1 - norm2) / (norm1 + norm2 + 1e-8))

                proximal_term = bhattacharyya_distance(local_probs,global_probs)
                loss = self.loss_func(log_probs, labels)+self.args.hy*proximal_term
                # if self.args.ja==1:
                # if iter == 0 and batch_idx == 0 :
                #     # 示例使用部分：计算 Hessian 矩阵
                #     layer_g = list(net.parameters())[-2]  # 基础层最后一层
                #     layer_l = list(net.parameters())[-1]  # 个性化层第一层
                #     # 计算 Hessian 矩阵
                #     # hessian_g, hessian_l = self.compute_hessian(net, dummy_data, dummy_target, layer_g, layer_l)
                #     hessian_g, hessian_l = self.compute_hessian(loss, layer_g, layer_l)
                #
                #     # 打印 Hessian 矩阵
                #     # print("Hessian for global layer:", hessian_g)
                #     # print("Hessian for local layer:", hessian_l)
                #     # 计算 Hessian 矩阵的 Frobenius 范数差异
                #     frobenius_difference = frobenius_norm_diff(hessian_g, hessian_l)
                #     print("Frobenius 范数差:", frobenius_difference)
                #     spectral_difference = spectral_norm_diff(hessian_g, hessian_l)
                #     print("谱范数差:", spectral_difference)
                #     # print("Frobenius norm difference between the Hessians:", frobenius_diff)
                #     frobenius_diff.backward(retain_graph=True)
                #     jacobian_optimizer.step()


                self.loss_func(log_probs, labels).backward(retain_graph=True)
                loss.backward()
                jacobian_optimizer.ascent_step()
                jacobian_optimizer.descent_step()
                # Descent Step
                # ds= self.loss_func(log_probs, labels)+self.args.hy*proximal_term





                optimizer.step()
                scheduler.step()
                total_local_probs += local_probs.sum(dim=0)

                batch_loss.append(loss.item())
                # print("loss: ", loss)
            epoch_loss.append(sum(batch_loss)/len(batch_loss))

        # DPnet = copy.deepcopy(net)
        # if self.args.dp_type != 0:
        #     with torch.no_grad():  # Ensure gradients are not computed for this operation
        #         for param in DPnet.parameters():
        #             param.data = add_noise(param.data, self.args.dp_type, self.args)
        sum_ = sum(total_local_probs)
        total_local_probs = torch.tensor([p/sum_ for p in total_local_probs])
        everyclient_distributed.append(total_local_probs)
        # accuracyafter, test_loss = test_img(DPnet, self.ldr_test.dataset, self.args)
        accuracybefor, test_loss1 = test_img(global_w, self.ldr_test.dataset, self.args)
        # print('accuracy',accuracybefor)
        # print('accuracy', accuracyafter)
        # print(f"batch_loss: {sum(batch_loss) / len(batch_loss)}, acc: {accuracy}, test_loss: {test_loss}", )
        # return net, sum(epoch_loss) / len(epoch_loss), scheduler.get_last_lr()[0],everyclient_distributed
        # print(copy.deepcopy(net))

        self.last_net = copy.deepcopy(net)
        # self.last_net = copy.deepcopy(DPnet)
        # return DPnet.state_dict(), sum(epoch_loss) / len(epoch_loss), scheduler.get_last_lr()[0],everyclient_distributed, accuracybefor
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss), scheduler.get_last_lr()[0],everyclient_distributed, accuracybefor

