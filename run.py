from os import name
from model.ViT import ViT
from model.VGG19 import VGG
from model.UNet import UNet
from model.ViT import ViTEmbedding

from data_setup import create_dataloader
from enegine import train

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch

from thop import profile, clever_format

import json
import numpy as np

def generate_orthogonal_vectors(num_vectors: int=4, dimension: int=768):
    """
    在指定维度的空间中生成指定数量的相互正交的随机向量。

    Args:
        num_vectors (int): 你想要生成的向量数量。
        dimension (int): 每个向量的维度。必须满足 dimension >= num_vectors。

    Returns:
        numpy.ndarray: 一个形状为 (num_vectors, dimension) 的数组，
                       每一行是一个标准正交向量（长度为1）。
    """

    random_matrix = np.random.randn(dimension, num_vectors)

    q_matrix, _ = np.linalg.qr(random_matrix)
    orthogonal_vectors = q_matrix.T

    return orthogonal_vectors

embedding_dim = 768
epochs = 100
lr = 1e-3
torch.cuda.set_device(0)
vector = generate_orthogonal_vectors()
vector = torch.tensor(vector)
vector = vector.cuda()
print(f"vector.shape{vector.shape}")

def Closs(
    y_pred: torch.Tensor, 
    y: torch.Tensor, 
    vectors : torch.Tensor = vector,
    margin: float = 0.5
):
    """
    自定义的对比损失函数，同时返回TP, FP, FN统计值。

    Args:
        y_pred (torch.Tensor): 模型的输出嵌入，形状为 (B, D)。
        y (torch.Tensor): 属性标签，形状为 (B, N)，值为0或1。
        vectors (torch.Tensor): 正交的原型向量，形状为 (N, D)。
        margin (float): 用于判断正负样本的余弦相似度阈值。

    Returns:
        tuple: 一个元组 (loss, tp, fp, fn)，包含标量损失和统计计数值。
    """


    device = y_pred.device
    y = y.to(device)
    vectors = vectors.to(device)
    similarities = F.cosine_similarity(y_pred.unsqueeze(1), vectors.unsqueeze(0), dim=2)
    cosine_distance = 1 - similarities
    loss = (y * cosine_distance.pow(2) + (1 - y) * F.relu(margin - cosine_distance).pow(2)).mean()

    predicted_labels = (similarities >= margin).float()

    tp = ((predicted_labels == 1) & (y == 1)).sum(dim=1)
    fp = ((predicted_labels == 1) & (y == 0)).sum(dim=1)
    fn = ((predicted_labels == 0) & (y == 1)).sum(dim=1)

    return loss, tp, fp, fn

def normal_loss(
    y_pred: torch.Tensor, 
    y: torch.Tensor
):
    """
    一个自定义损失函数，用于处理16分类预测与4位二进制属性标签。

    该函数执行以下操作：
    1. 将4位二进制的真实标签 y 转换为0-15的整数类别。
    2. 使用交叉熵计算模型16分类预测 y_pred 与转换后的整数标签之间的损失。
    3. 从 y_pred 中获取预测类别，并将其解码为4位二进制预测属性。
    4. 比较预测属性与真实属性 y，计算整个批次的 TP, FP, FN。

    Args:
        y_pred (torch.Tensor): 模型的softmax输出，形状为 (B, 16)。
        y (torch.Tensor): 真实的二进制属性标签，形状为 (B, 4)，值为0或1。

    Returns:
        tuple: 一个元组 (loss, tp, fp, fn)，包含标量损失和整个批次的统计计数值。
    """

    device = y_pred.device
    y = y.to(device)
    powers_of_2 = torch.tensor([8, 4, 2, 1], device=device, dtype=torch.long)
    target_indices = (y.long() * powers_of_2).sum(dim=1)

    # 使用交叉熵损失函数
    criterion = nn.CrossEntropyLoss()
    loss = criterion(y_pred, target_indices)

    predicted_indices = torch.argmax(y_pred, dim=1)

    predicted_binary_labels = torch.stack([
        (predicted_indices >> 3) & 1,  # 第3位 (最高位)
        (predicted_indices >> 2) & 1,  # 第2位
        (predicted_indices >> 1) & 1,  # 第1位
        (predicted_indices >> 0) & 1   # 第0位 (最低位)
    ], dim=1).float()

    tp = ((predicted_binary_labels == 1) & (y == 1)).sum()
    fp = ((predicted_binary_labels == 1) & (y == 0)).sum()
    fn = ((predicted_binary_labels == 0) & (y == 1)).sum()

    return loss, tp, fp, fn


train_dataloader , test_dataloader , _ = create_dataloader(batch_size = 32)

model_name = "ViTEmbedding"

if model_name == "VGG":
    model = VGG()
    loss_function = normal_loss
elif model_name == "ViT":
    model = ViT()
    loss_function = normal_loss
elif model_name == "UNet":
    model = UNet()
    loss_function = normal_loss
elif model_name == "ViTEmbedding":
    model = ViTEmbedding()
    loss_function = Closs

model = model.cuda()
optimizer = optim.Adam(model.parameters() , lr = lr)

input_tensor = torch.randn(32, 3, 224, 224)

macs, params = profile(model, inputs=(input_tensor,))

print(f"原始 MACs: {macs}") 
print(f"原始参数量: {params}")

results = train(model=model, loss_fn = loss_function,
                train_dataloader=train_dataloader , test_dataloader=test_dataloader,
                optimizer = optimizer,
                epochs = epochs,
                device = torch.device('cuda:0'))

with open("./logs/results.json", "w") as f:
    json.dump(results, f, indent=4)

