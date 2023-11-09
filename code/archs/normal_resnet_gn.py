'''ResNet in PyTorch.
For Pre-activation ResNet, see 'preact_resnet.py'.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
from .custom_modules import SequentialWithArgs

from .normal_resnet import Bottleneck, ResNet

# define class BottleneckIN from Bottleneck, which replace BatchNorm2d in Bottleneck with GroupNorm
class BottleneckGN(Bottleneck):
    def __init__(self, in_planes, planes, stride=1, groups=1):
        super(BottleneckGN, self).__init__(in_planes, planes, stride)
        self.bn1 = nn.GroupNorm(groups, planes, affine=True)
        self.bn2 = nn.GroupNorm(groups, planes, affine=True)
        self.bn3 = nn.GroupNorm(groups, self.expansion*planes, affine=True)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(groups, self.expansion*planes, affine=True)
            )

# define class ResNetIN from ResNet, which replace BatchNorm2d in ResNet with GroupNorm
class ResNetGN(ResNet):
    def __init__(self, block, num_blocks, num_classes=10, feat_scale=1, wm=1, groups=1):
        self.groups = groups
        super(ResNetGN, self).__init__(block, num_blocks, num_classes, feat_scale, wm)
        self.bn1 = nn.GroupNorm(groups, self.first_planes, affine=True)
        
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, groups=self.groups))
            self.in_planes = planes * block.expansion
        return SequentialWithArgs(*layers)
    

def ResNet152GN(**kwargs):
    return ResNetGN(BottleneckGN, [3,8,36,3], **kwargs)

resnet152gn = ResNet152GN


# resnet18thin = ResNet18Thin
if __name__ == "__main__":
    net = resnet152gn()
    y = net(torch.randn(2,3,32,32))
    print(y.size())