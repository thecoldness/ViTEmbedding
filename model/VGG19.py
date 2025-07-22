import torch
import torch.nn as nn
from typing import Union, List, Dict, Any, cast

vgg19_config: List[Union[str, int]] = [
    64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 
    512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'
]

class VGG(nn.Module):
    """
    VGG模型通用实现
    """
    def __init__(
        self, 
        features: nn.Module, 
        num_classes: int = 16, 
        init_weights: bool = True
    ) -> None:
        """
        初始化VGG模型
        
        Args:
            features (nn.Module): 卷积特征提取器部分
            num_classes (int): 分类器的输出类别数，默认为1000 (ImageNet)
            init_weights (bool): 是否对模型进行权重初始化
        """
        super(VGG, self).__init__()
        self.features = features
        # 自适应平均池化层，可以将任意大小的特征图输出为固定的 7x7 大小
        # 这使得模型可以接受不同尺寸的输入图像
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes),
            nn.Softmax(dim=1)  # 输出概率分布
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        # 将多维特征图展平为一维向量，以输入到全连接层
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # 使用Kaiming He初始化方法，适用于ReLU激活函数
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)