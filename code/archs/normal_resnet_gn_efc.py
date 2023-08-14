'''ResNet in PyTorch.
For Pre-activation ResNet, see 'preact_resnet.py'.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch.nn as nn
import torch.nn.functional as F

from .normal_resnet_gn import ResNetGN, BottleneckGN

    
# define class ResNetGNEfc from ResNetGN which add an extra fc layer
class ResNetGNEfc(ResNetGN):
    def __init__(self, block, num_blocks, num_classes=10, feat_scale=1, wm=1, groups=1, extra_fc_dim=2048):
        super(ResNetGNEfc, self).__init__(block, num_blocks, num_classes, feat_scale, wm, groups)
        self.extra_fc = nn.Linear(2048, extra_fc_dim)

    def forward(self, x, clean_bs=None):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.adaptive_avg_pool2d(out, 1)
        pre_out = out.view(out.size(0), -1)
        if clean_bs is None:
            final = self.linear(pre_out)
            return final
        else:
            clean_out = pre_out[:clean_bs]
            extra_out = pre_out[clean_bs:]
            final = self.linear(clean_out)
            extra_final = self.extra_fc(extra_out)
            return final, extra_final        


def ResNet152GNEfc(**kwargs):
    return ResNetGNEfc(BottleneckGN, [3,8,36,3], **kwargs)

resnet152gnefc = ResNet152GNEfc
