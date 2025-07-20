from model.ViT import ViT
from data_setup import create_dataloader
from enegine import train

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch


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
    if num_vectors > dimension:
        raise ValueError(
            f"无法在 {dimension} 维空间中找到 {num_vectors} 个相互正交的向量。"
            f"向量的数量必须小于或等于其维度。"
        )

    # 1. 创建一个随机的 (dimension x num_vectors) 矩阵。
    # 使用 np.random.randn 从标准正态分布中采样，这在数学上性质更好。
    random_matrix = np.random.randn(dimension, num_vectors)

    # 2. 对这个矩阵进行QR分解。
    # np.linalg.qr 返回一个元组 (Q, R)。我们只需要正交矩阵 Q。
    q_matrix, _ = np.linalg.qr(random_matrix)

    # q_matrix 的形状是 (dimension, num_vectors)，其列向量是标准正交的。
    # 为了方便使用，我们将其转置，使得每个向量成为一个行向量。
    # 最终返回的数组形状为 (num_vectors, dimension)。
    orthogonal_vectors = q_matrix.T

    return orthogonal_vectors

embedding_dim = 768
epochs = 100
lr = 1e-3
vector = generate_orthogonal_vectors()
vector = torch.tensor(vector)
vector = vector.cuda()
print(f"vector.shape{vector.shape}")

def Closs(y_pred: torch.Tensor, y: torch.Tensor):
    """
    自定义的对比损失函数。

    Args:
        y_pred (torch.Tensor): 模型的输出嵌入，形状为 (batch_size, embedding_dim)。
        y (torch.Tensor): 属性标签，形状为 (batch_size, num_attributes)，值为0或1。
        vectors (torch.Tensor): 正交的原型向量，形状为 (num_attributes, embedding_dim)。

    Returns:
        torch.Tensor: 计算出的标量损失值。
    """
    # 确保输入在同一个设备上
    device = y_pred.device
    y = y.to(device)

    # 1. 计算 y_pred 中每个向量与所有原型向量的余弦相似度
    # y_pred: (B, D) -> (B, 1, D)
    # vectors: (N, D) -> (1, N, D)
    # F.cosine_similarity 计算后，得到 (B, N) 的相似度矩阵
    # B = batch_size, D = dimension (768), N = num_attributes (4)
    similarities = F.cosine_similarity(y_pred.unsqueeze(1), vector.unsqueeze(0), dim=2)

    # 2. 计算正样本对的损失
    # 当 y == 1 时，我们希望 similarity -> 1。损失为 1 - similarity。
    # 我们用 y 作为掩码，只在 y==1 的位置计算损失。
    positive_loss = (1 - similarities) * y

    # 3. 计算负样本对的损失
    # 当 y == 0 时，我们希望 similarity -> 0 或负数。损失为 max(0, similarity)。
    # 我们用 (1 - y) 作为掩码，只在 y==0 的位置计算损失。
    negative_loss = F.relu(similarities) * (1 - y)

    # 4. 合并总损失
    # 将正负样本的损失相加，得到每个样本-属性对的损失矩阵
    total_loss_matrix = positive_loss + negative_loss

    # 5. 计算整个批次的平均损失
    # .mean() 会计算所有元素 (B * N) 的平均值
    loss = total_loss_matrix.mean()

    return loss

train_dataloader , test_dataloader , _ = create_dataloader(batch_size = 64)

model = ViT()

model = model.cuda()
optimizer = optim.Adam(model.parameters() , lr = lr)

train(model=model, loss_fn = Closs,
      train_dataloader=train_dataloader , test_dataloader=test_dataloader,
      optimizer = optimizer,
      epochs = epochs,
      device = torch.device('cuda:0'))