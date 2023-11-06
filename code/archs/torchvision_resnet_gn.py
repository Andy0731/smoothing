'''ResNet in PyTorch.
For Pre-activation ResNet, see 'preact_resnet.py'.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
from typing import Callable, Optional

import torch
import torch.nn as nn

from torchvision.models.resnet import ResNet as torchvision_resnet
from torchvision.models.resnet import Bottleneck as torchvision_bottleneck
from torchvision.models.resnet import conv1x1


class BottleneckTVGN(torchvision_bottleneck):
    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        gn_groups: int = 1,
    ) -> None:
        super(BottleneckTVGN, self).__init__(inplanes, planes, stride, downsample, groups, base_width, dilation, norm_layer)
        width = int(planes * (base_width / 64.0)) * groups
        self.bn1 = nn.GroupNorm(gn_groups, width, affine=True)
        self.bn2 = nn.GroupNorm(gn_groups, width, affine=True)
        self.bn3 = nn.GroupNorm(gn_groups, planes * self.expansion, affine=True) 


class ResNetTVGN(torchvision_resnet):
    def __init__(self, block, layers, num_classes=1000, gn_groups=1):
        self.gn_groups = gn_groups
        super(ResNetTVGN, self).__init__(block, layers, num_classes)
        self.bn1 = nn.GroupNorm(gn_groups, 64, affine=True)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.GroupNorm(self.gn_groups, planes * block.expansion, affine=True),
            )
        
        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer, gn_groups=self.gn_groups
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    gn_groups=self.gn_groups,
                )
            )

        return nn.Sequential(*layers)    


def resnet152tvgn(**kwargs):
    return ResNetTVGN(BottleneckTVGN, [3,8,36,3], **kwargs)


# resnet18thin = ResNet18Thin
if __name__ == "__main__":
    net = resnet152tvgn(gn_groups=32)
    print(net)
    y = net(torch.randn(2,3,224,224))
    print(y.size())
