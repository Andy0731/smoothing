'''ResNet in PyTorch.
For Pre-activation ResNet, see 'preact_resnet.py'.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn

from .normal_resnet import Bottleneck, ResNet

# define class BottleneckIN from Bottleneck, which replace BatchNorm2d in Bottleneck with InstanceNorm2d
class BottleneckIN(Bottleneck):
    def __init__(self, in_planes, planes, stride=1):
        super(BottleneckIN, self).__init__(in_planes, planes, stride)
        self.bn1 = nn.InstanceNorm2d(planes, affine=True)
        self.bn2 = nn.InstanceNorm2d(planes, affine=True)
        self.bn3 = nn.InstanceNorm2d(self.expansion*planes, affine=True)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.InstanceNorm2d(self.expansion*planes, affine=True)
            )

# define class ResNetIN from ResNet, which replace BatchNorm2d in ResNet with InstanceNorm2d
class ResNetIN(ResNet):
    def __init__(self, block, num_blocks, num_classes=10, feat_scale=1, wm=1):
        super(ResNetIN, self).__init__(block, num_blocks, num_classes, feat_scale, wm)
        self.bn1 = nn.InstanceNorm2d(self.first_planes, affine=True)

def ResNet152IN(**kwargs):
    return ResNetIN(BottleneckIN, [3,8,36,3], **kwargs)

resnet152in = ResNet152IN


# resnet18thin = ResNet18Thin
if __name__ == "__main__":
    net = resnet152in()
    y = net(torch.randn(2,3,32,32))
    print(y.size())
