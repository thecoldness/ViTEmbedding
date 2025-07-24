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
# print(f"vector.shape{vector.shape}")

def Closs(
    y_pred: torch.Tensor, 
    y: torch.Tensor, 
    vectors : torch.Tensor = vector,
    margin: float = 0.5
):


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

    device = y_pred.device
    y = y.to(device)
    powers_of_2 = torch.tensor([8, 4, 2, 1], device=device, dtype=torch.long)
    target_indices = (y.long() * powers_of_2).sum(dim=1)

    criterion = nn.CrossEntropyLoss()
    loss = criterion(y_pred, target_indices)

    predicted_indices = torch.argmax(y_pred, dim=1)

    predicted_binary_labels = torch.stack([
        (predicted_indices >> 3) & 1,  
        (predicted_indices >> 2) & 1,  
        (predicted_indices >> 1) & 1,
        (predicted_indices >> 0) & 1 
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

print(f"MACs: {macs}") 
print(f"parameters amount: {params}")

results = train(model=model, loss_fn = loss_function,
                train_dataloader=train_dataloader , test_dataloader=test_dataloader,
                optimizer = optimizer,
                epochs = epochs,
                device = torch.device('cuda:0'))

with open("./logs/results.json", "w") as f:
    json.dump(results, f, indent=4)

