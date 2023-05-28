'''ResNet in PyTorch.
For Pre-activation ResNet, see 'preact_resnet.py'.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from archs.normal_resnet_nemb import TimestepBlock, TimestepEmbedSequential
from archs.normal_resnet import Bottleneck
from archs.normal_resnet_nemb import ResNetNemb


class BottleneckEmb(Bottleneck, TimestepBlock):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super().__init__(in_planes, planes, stride=stride)

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(64, planes)
        )

    def forward(self, x, emb):
        out = F.relu(self.bn1(self.conv1(x)))

        emb_out = self.emb_layers(emb) # (N, planes)
        out = out + emb_out[:, :, None, None] # (N, planes, 1, 1)

        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        return F.relu(out)
    

class ResNetBlkEmb(ResNetNemb):
    def __init__(self, block, num_blocks, num_classes=10, feat_scale=1, wm=1, nemb_layer='all'):
        super().__init__(block, num_blocks, num_classes=num_classes, feat_scale=feat_scale, wm=wm, nemb_layer=nemb_layer)
        all_layers = [self.layer1, self.layer2, self.layer3, self.layer4]
        all_blocks = []
        for layer in all_layers:
            for block in layer:
                all_blocks.append(block)
        self.all_blocks = TimestepEmbedSequential(*all_blocks)

    def forward(self, x, noise_sd):
        # noise_sd embed
        bt_noise_sd = torch.ones(x.size(0)).to(x.device) * noise_sd # (N)
        bt_noise_sd = torch.floor(bt_noise_sd * 1000) # (N)
        nemd = self.noise_sd_embed(bt_noise_sd, self.emb_dim) # (N, 32)
        nemd = self.emb_mlp(nemd) # (N, 64)

        # init conv on x
        out = F.relu(self.bn1(self.conv1(x))) # (N, 64, 32, 32)
        out = out + nemd.view(-1, 64, 1, 1) # (N, 64, 32, 32)

        # blocks
        for block in self.all_blocks:
            out = block(out, nemd)

        out = F.adaptive_avg_pool2d(out, 1)
        pre_out = out.view(out.size(0), -1)
        final = self.linear(pre_out)
        return final


def ResNetBlkEmb152(**kwargs):
    return ResNetBlkEmb(BottleneckEmb, [3,8,36,3], **kwargs)

resnetblkemd152 = ResNetBlkEmb152