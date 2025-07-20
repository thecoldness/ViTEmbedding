from model.ViT import ViT
from data_setup import create_dataloaders
from enegine import train

train_dataloader , test_dataloader , _ = create_dataloaders(batch_size = 64)

