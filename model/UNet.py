import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class DoubleConv(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, mid_channels: int = None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)


class UNet(nn.Module):
    """
    一个带有分类头的UNet模型，用于同时进行图像分割和图像分类。
    """
    def __init__(self, 
                 n_channels: int,      # 输入图像的通道数 (例如 RGB为3)
                 n_seg_classes: int,   # 分割任务的类别数
                 n_img_classes: int=16,   # 分类任务的类别数
                 bilinear: bool = True # 是否使用双线性插值进行上采样
                ):
        """
        初始化模型
        
        Args:
            n_channels (int): 输入通道数.
            n_seg_classes (int): 分割输出的类别数.
            n_img_classes (int): 图像分类输出的类别数.
            bilinear (bool): 为True时使用双线性插值上采样，否则使用转置卷积.
        """
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_seg_classes = n_seg_classes
        self.n_img_classes = n_img_classes
        self.bilinear = bilinear
        factor = 2 if bilinear else 1

        # --- 编码器 (Contracting Path) ---
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = DoubleConv(64, 128)
        self.down2 = DoubleConv(128, 256)
        self.down3 = DoubleConv(256, 512)
        self.down4 = DoubleConv(512, 1024 // factor)
        
        # --- 分类头 (Classification Head) ---
        # 连接在编码器最深处，即瓶颈处
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)), # 全局平均池化
            nn.Flatten(),
            nn.Linear(1024 // factor, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, n_img_classes),
            nn.Softmax(dim=1)
        )

        # --- 解码器 (Expansive Path) ---
        self.up1 = nn.ConvTranspose2d(1024 // factor, 512 // factor, kernel_size=2, stride=2)
        self.conv1 = DoubleConv(1024, 512 // factor)
        
        self.up2 = nn.ConvTranspose2d(512 // factor, 256 // factor, kernel_size=2, stride=2)
        self.conv2 = DoubleConv(512, 256 // factor)

        self.up3 = nn.ConvTranspose2d(256 // factor, 128 // factor, kernel_size=2, stride=2)
        self.conv3 = DoubleConv(256, 128 // factor)

        self.up4 = nn.ConvTranspose2d(128 // factor, 64, kernel_size=2, stride=2)
        self.conv4 = DoubleConv(128, 64)
        
        # --- 分割输出头 ---
        self.outc = nn.Conv2d(64, n_seg_classes, kernel_size=1)


    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播，返回分割图和分类结果
        """
        # --- 编码器 ---
        x1 = self.inc(x)
        x2 = self.down1(F.max_pool2d(x1, 2))
        x3 = self.down2(F.max_pool2d(x2, 2))
        x4 = self.down3(F.max_pool2d(x3, 2))
        bottleneck = self.down4(F.max_pool2d(x4, 2))

        # 从瓶颈特征中提取全局信息进行分类
        classification_output = self.classifier(bottleneck)
        
        u1 = self.up1(bottleneck)
        u1 = torch.cat([x4, u1], dim=1)
        u1 = self.conv1(u1)

        u2 = self.up2(u1)
        u2 = torch.cat([x3, u2], dim=1)
        u2 = self.conv2(u2)

        u3 = self.up3(u2)
        u3 = torch.cat([x2, u3], dim=1)
        u3 = self.conv3(u3)

        u4 = self.up4(u3)
        u4 = torch.cat([x1, u4], dim=1)
        u4 = self.conv4(u4)

        segmentation_output = self.outc(u4)

        return classification_output