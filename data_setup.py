import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from typing import List

# -----------------------------------------------------------------------------
# 1. 自定义数据集类 (处理底层数据加载)
# -----------------------------------------------------------------------------
class CelebAAttributeDataset(Dataset):
    """
    自定义CelebA属性数据集。
    根据分区（train/val/test）和指定的属性名称加载数据。
    """
    def __init__(self, root_dir: str, partition: str, transform=None, selected_attrs: List[str] = None):
        self.root_dir = root_dir
        self.transform = transform
        self.partition_map = {'train': 0, 'val': 1, 'test': 2}
        
        if partition not in self.partition_map:
            raise ValueError(f"Partition must be one of {list(self.partition_map.keys())}")
            
        self.selected_partition_code = self.partition_map[partition]
        
        # 文件路径
        self.img_dir = os.path.join(self.root_dir, 'Img', 'img_align_celeba')
        attr_file_path = os.path.join(self.root_dir, 'Anno', 'list_attr_celeba.txt')
        partition_file_path = os.path.join(self.root_dir, 'Eval', 'list_eval_partition.txt')
        
        # 解析和筛选数据
        self._prepare_data(attr_file_path, partition_file_path, selected_attrs)

    def _prepare_data(self, attr_file_path, partition_file_path, selected_attrs):
        # 解析属性文件
        with open(attr_file_path, 'r') as f:
            lines = f.readlines()
        
        self.all_attribute_names = lines[1].strip().split()
        attr_name_to_idx = {name: i for i, name in enumerate(self.all_attribute_names)}

        # 解析分区文件
        with open(partition_file_path, 'r') as f:
            partition_lines = f.readlines()

        self.samples = []
        for i, line in enumerate(lines[2:]): # 属性从第3行开始
            parts = line.strip().split()
            filename = parts[0]
            
            # 检查该图片是否属于当前分区
            # CelebA的三个文件行数一一对应，所以可以直接用行号i
            p_filename, p_code = partition_lines[i].strip().split()
            if filename == p_filename and int(p_code) == self.selected_partition_code:
                # 将属性从 '-1'/'1' 转换为 0/1
                attrs_01 = [(int(v) + 1) // 2 for v in parts[1:]]
                self.samples.append((filename, torch.tensor(attrs_01, dtype=torch.float32)))
        
        # 如果指定了要选择的属性，则进行筛选
        if selected_attrs:
            try:
                # 获取指定属性名称对应的索引
                selected_indices = [attr_name_to_idx[name] for name in selected_attrs]
            except KeyError as e:
                raise ValueError(f"属性 '{e.args[0]}' 不存在于数据集中。可用属性为: {self.all_attribute_names}") from e
            
            # 对每个样本的标签进行筛选
            self.samples = [(fname, labels[selected_indices]) for fname, labels in self.samples]
            self.selected_attribute_names = selected_attrs
        else:
            self.selected_attribute_names = self.all_attribute_names


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        filename, labels = self.samples[idx]
        img_path = os.path.join(self.img_dir, filename)
        
        # x 为图片
        x = Image.open(img_path).convert('RGB')
        
        # y 为属性标签
        y = labels
        
        if self.transform:
            x = self.transform(x)
            
        return x, y

# -----------------------------------------------------------------------------
# 2. 主函数 (封装所有逻辑)
# -----------------------------------------------------------------------------
def get_celeba_dataloaders(
    root_dir: str, 
    batch_size: int, 
    selected_attrs: List[str], 
    img_size: int = 224, 
    num_workers: int = 4
):
    """
    创建并返回CelebA的训练集和测试集的DataLoader。

    Args:
        root_dir (str): CelebA数据集的根目录。
        batch_size (int): 批次大小。
        selected_attrs (List[str]): 一个包含所需属性名称的字符串列表。
        img_size (int, optional): 图像将被调整到的尺寸。默认为 224。
        num_workers (int, optional): 用于数据加载的子进程数。默认为 4。

    Returns:
        tuple: 一个包含 (train_dataloader, test_dataloader, attribute_names) 的元组。
    """
    # 定义图像预处理流程
    image_transforms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    # 创建训练集 Dataset
    train_dataset = CelebAAttributeDataset(
        root_dir=root_dir,
        partition='train',
        transform=image_transforms,
        selected_attrs=selected_attrs
    )

    # 创建测试集 Dataset
    test_dataset = CelebAAttributeDataset(
        root_dir=root_dir,
        partition='test',
        transform=image_transforms,
        selected_attrs=selected_attrs
    )
    
    # 创建训练集 DataLoader
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True, # 丢弃最后一个不完整的batch
    )

    # 创建测试集 DataLoader
    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False, # 测试集不需要打乱
        num_workers=num_workers,
        pin_memory=True,
    )

    print(f"成功创建DataLoader。")
    print(f"训练集样本数: {len(train_dataset)}, 测试集样本数: {len(test_dataset)}")
    print(f"选取的属性: {train_dataset.selected_attribute_names}")
    
    return train_dataloader, test_dataloader, train_dataset.selected_attribute_names

# -----------------------------------------------------------------------------
# 3. 使用示例
# -----------------------------------------------------------------------------
if __name__ == '__main__':
    # --- 配置 ---
    # !!! 修改为你自己的CelebA根目录 !!!
    CELEBA_ROOT = './CelebA'  
    BATCH_SIZE = 64
    
    # 选择你感兴趣的属性，可以直接使用名称
    ATTRIBUTES_TO_SELECT = [
        'Smiling', 
        'Male', 
        'Young', 
        'Wearing_Hat',
        'Heavy_Makeup'
    ]
    
    # --- 调用函数获取Dataloaders ---
    try:
        train_loader, test_loader, attr_names = get_celeba_dataloaders(
            root_dir=CELEBA_ROOT,
            batch_size=BATCH_SIZE,
            selected_attrs=ATTRIBUTES_TO_SELECT
        )
        
        # --- 演示如何使用返回的DataLoader ---
        print("\n--- 演示从 train_loader 中取出一个批次 ---")
        
        # x是图片, y是属性
        x_batch, y_batch = next(iter(train_loader))
        
        print(f"图片批次 (x) 的形状: {x_batch.shape}")
        print(f"属性标签批次 (y) 的形状: {y_batch.shape}")
        
        print("\n选取的属性顺序为:", attr_names)
        print(f"第一个样本的属性 (y[0]): {y_batch[0]}")
        
    except FileNotFoundError:
        print(f"\n!!! 错误: 找不到数据集文件。请确保'{CELEBA_ROOT}'路径正确，")
        print("并且该路径下包含 'Anno', 'Eval', 'Img' 三个文件夹。")
    except ValueError as e:
        print(f"\n!!! 错误: {e}")