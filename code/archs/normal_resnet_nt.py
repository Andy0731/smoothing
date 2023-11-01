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
class BottleneckNT(Bottleneck):
    def __init__(self, in_planes, planes, stride=1, track_running_stats=True):
        super(BottleneckNT, self).__init__(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, track_running_stats=track_running_stats)
        self.bn2 = nn.BatchNorm2d(planes, track_running_stats=track_running_stats)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes, track_running_stats=track_running_stats)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes, track_running_stats=track_running_stats)
            )

# define class ResNetIN from ResNet, which replace BatchNorm2d in ResNet with GroupNorm
class ResNetNT(ResNet):
    def __init__(self, block, num_blocks, num_classes=10, feat_scale=1, wm=1, track_running_stats=True):
        self.track_running_stats = track_running_stats
        super(ResNetNT, self).__init__(block, num_blocks, num_classes, feat_scale, wm)
        self.bn1 = nn.BatchNorm2d(self.first_planes, track_running_stats=self.track_running_stats)
        
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, track_running_stats=self.track_running_stats))
            self.in_planes = planes * block.expansion
        return SequentialWithArgs(*layers)
    

def ResNet152NT(**kwargs):
    return ResNetNT(BottleneckNT, [3,8,36,3], **kwargs)

resnet152nt = ResNet152NT


# resnet18thin = ResNet18Thin
if __name__ == "__main__":
    net = resnet152nt()
    print(net)
    y = net(torch.randn(2,3,32,32))
    print(y.size())
