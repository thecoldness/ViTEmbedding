import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from typing import List

class CelebAAttributeDataset(Dataset):
    def __init__(self, root_dir: str, partition: str, transform=None, selected_attrs: List[str] = None):
        self.root_dir = root_dir
        self.transform = transform
        self.partition_map = {'train': 0, 'val': 1, 'test': 2}
        
        if partition not in self.partition_map:
            raise ValueError(f"Partition must be one of {list(self.partition_map.keys())}")
            
        self.selected_partition_code = self.partition_map[partition]
        
        self.img_dir = os.path.join(self.root_dir, 'Img', 'img_align_celeba')
        attr_file_path = os.path.join(self.root_dir, 'Anno', 'list_attr_celeba.txt')
        partition_file_path = os.path.join(self.root_dir, 'Eval', 'list_eval_partition.txt')
        
        self._prepare_data(attr_file_path, partition_file_path, selected_attrs)

    def _prepare_data(self, attr_file_path, partition_file_path, selected_attrs):
        with open(attr_file_path, 'r') as f:
            lines = f.readlines()
        
        self.all_attribute_names = lines[1].strip().split()
        attr_name_to_idx = {name: i for i, name in enumerate(self.all_attribute_names)}

        with open(partition_file_path, 'r') as f:
            partition_lines = f.readlines()

        self.samples = []
        for i, line in enumerate(lines[2:]): # 属性从第3行开始
            parts = line.strip().split()
            filename = parts[0]
            
            p_filename, p_code = partition_lines[i].strip().split()
            if filename == p_filename and int(p_code) == self.selected_partition_code:
                # 将属性从 '-1'/'1' 转换为 0/1
                attrs_01 = [(int(v) + 1) // 2 for v in parts[1:]]
                if sum(attrs_01) > 0:
                    self.samples.append((filename, torch.tensor(attrs_01, dtype=torch.float32)))

        if selected_attrs:
            selected_indices = [attr_name_to_idx[name] for name in selected_attrs]
            
            self.samples = [(fname, labels[selected_indices]) for fname, labels in self.samples]
            self.selected_attribute_names = selected_attrs
        else:
            self.selected_attribute_names = self.all_attribute_names


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        filename, labels = self.samples[idx]
        img_path = os.path.join(self.img_dir, filename)
        
        x = Image.open(img_path).convert('RGB')
        
        y = labels
        
        if self.transform:
            x = self.transform(x)
            
        return x, y

def create_dataloader(
    batch_size: int, 
    img_size: int = 224, 
    num_workers: int = 4
):
    """

    Args:
        root_dir (str): CelebA数据集的根目录。
        batch_size (int): 批次大小。
        selected_attrs (List[str]): 一个包含所需属性名称的字符串列表。
        img_size (int, optional): 图像将被调整到的尺寸。默认为 224。
        num_workers (int, optional): 用于数据加载的子进程数。默认为 4。

    Returns:
        tuple: 一个包含 (train_dataloader, test_dataloader, attribute_names) 的元组。
    """
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) 

    root_dir = os.path.join(SCRIPT_DIR, 'dataset', 'CelebA')
    selected_attrs = [
        'Bangs', 
        'Eyeglasses', 
        'Goatee', 
        'Mustache',
    ]
    image_transforms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    train_dataset = CelebAAttributeDataset(
        root_dir=root_dir,
        partition='train',
        transform=image_transforms,
        selected_attrs=selected_attrs
    )

    test_dataset = CelebAAttributeDataset(
        root_dir=root_dir,
        partition='test',
        transform=image_transforms,
        selected_attrs=selected_attrs
    )
    
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    return train_dataloader, test_dataloader, train_dataset.selected_attribute_names
