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

    random_matrix = np.random.randn(dimension, num_vectors)

    q_matrix, _ = np.linalg.qr(random_matrix)
    orthogonal_vectors = q_matrix.T

    return orthogonal_vectors

embedding_dim = 768
epochs = 100
lr = 1e-3
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
    # --- 损失计算部分 (与之前保持不变) ---
    device = y_pred.device
    y = y.to(device)
    vectors = vectors.to(device)

    # 1. 计算余弦相似度矩阵
    # B = batch_size, D = dimension, N = num_attributes
    similarities = F.cosine_similarity(y_pred.unsqueeze(1), vectors.unsqueeze(0), dim=2) # Shape: (B, N)

    # 2. 计算损失
    positive_loss = (1 - similarities) * y
    negative_loss = F.relu(similarities) * (1 - y)
    total_loss_matrix = positive_loss + negative_loss
    loss = total_loss_matrix.mean()

    # --- 新增：TP, FP, FN 统计部分 ---

    # 3. 根据 margin 生成预测标签
    # 如果 similarity >= margin，则预测为1，否则为0
    predicted_labels = (similarities >= margin).float() # Shape: (B, N)

    # 4. 计算统计指标
    # 使用布尔运算和 .sum() 来高效地计数
    # TP: 预测为1且真实为1
    tp = ((predicted_labels == 1) & (y == 1)).sum(dim=1)
    
    # FP: 预测为1但真实为0
    fp = ((predicted_labels == 1) & (y == 0)).sum(dim=1)
    
    # FN: 预测为0但真实为1
    fn = ((predicted_labels == 0) & (y == 1)).sum(dim=1)
    
    # (可选) TN: 预测为0且真实为0
    # tn = ((predicted_labels == 0) & (y == 0)).sum().item()

    return loss, tp, fp, fn

train_dataloader , test_dataloader , _ = create_dataloader(batch_size = 64)

model = ViT()

model = model.cuda()
optimizer = optim.Adam(model.parameters() , lr = lr)

results = train(model=model, loss_fn = Closs,
                train_dataloader=train_dataloader , test_dataloader=test_dataloader,
                optimizer = optimizer,
                epochs = epochs,
                device = torch.device('cuda:0'))

