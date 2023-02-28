import torch
import torch.nn as nn
import torch.nn.functional as F
from .normal_resnet import ResNet, Bottleneck


class ResNetAvgn(ResNet):
    def __init__(self, block, num_blocks, num_classes=10, feat_scale=1, wm=1, avgn_loc=None, avgn_num=1):
        super().__init__(block, num_blocks, num_classes=10, feat_scale=1, wm=1)
        self.avgn_loc = avgn_loc
        self.avgn_num = avgn_num

    def forward(self, x, with_latent=False, fake_relu=False, no_relu=False):
        assert (not no_relu),  \
            "no_relu not yet supported for this architecture"
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out, fake_relu=fake_relu)
        out = F.avg_pool2d(out, 4)
        pre_out = out.view(out.size(0), -1)
        if self.avgn_loc == 'before_fc': # bxn,c
            bxn, c = pre_out.shape
            pre_out = pre_out.view(-1, self.avgn_num, c) # b,n,c
            pre_out = pre_out.mean(dim=1) # b,c
        final = self.linear(pre_out)
        if self.avgn_loc == 'after_fc': # bxn,c
            bxn, c = final.shape
            final = final.view(-1, self.avgn_num, c) # b,n,c
            final = final.mean(dim=1) # b,c            
        if with_latent:
            return final, pre_out
        return final


def ResNet152Avgn(**kwargs):
    return ResNetAvgn(Bottleneck, [3,8,36,3], **kwargs)  


resnet152avgn = ResNet152Avgn

