import torch.nn as nn
from .normal_resnet import ResNet, Bottleneck


class ResNetNconv(ResNet):
    def __init__(self, block, num_blocks, num_classes=10, feat_scale=1, wm=1, avgn_num=1):
        super().__init__(block, num_blocks, num_classes=10, feat_scale=1, wm=1)
        self.avgn_num = avgn_num
        self.conv1 = nn.Conv2d(3*avgn_num, 64, kernel_size=3, stride=1,
                        padding=1, bias=False)
        print('model conv1 in_channels ', 3*avgn_num, ' out_channels ', 64)
        

def ResNet152Nconv(**kwargs):
    return ResNetNconv(Bottleneck, [3,8,36,3], **kwargs)


resnet152nconv = ResNet152Nconv