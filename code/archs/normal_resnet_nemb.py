'''ResNet in PyTorch.
For Pre-activation ResNet, see 'preact_resnet.py'.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from .normal_resnet import ResNet, Bottleneck
from improved_diffusion.nn import timestep_embedding
from abc import abstractmethod


class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, emb):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x


class ResNetNemb(ResNet, TimestepBlock):
    def __init__(self, block, num_blocks, num_classes=10, feat_scale=1, wm=1, nemb_layer='frs'):
        super().__init__(block, num_blocks, num_classes=num_classes, feat_scale=feat_scale, wm=wm)
        self.noise_sd_embed = timestep_embedding
        self.emb_dim = 32
        self.model_dim = 64
        self.emb_mlp = nn.Sequential(
            nn.Linear(self.emb_dim, self.model_dim),
            nn.SiLU(),
            nn.Linear(self.model_dim, self.model_dim),
        )
        self.nemb_layer = nemb_layer

    def forward(self, x, noise_sd):
        out = F.relu(self.bn1(self.conv1(x))) # (N, 64, 32, 32)


        bt_noise_sd = torch.ones(x.size(0)).to(x.device) * noise_sd # (N)
        bt_noise_sd = torch.floor(bt_noise_sd * 1000) # (N)
        nemd = self.noise_sd_embed(bt_noise_sd, self.emb_dim) # (N, 32)
        nemd = self.emb_mlp(nemd) # (N, 64)

        if self.nemb_layer == 'frs':
            out = out + nemd.view(-1, 64, 1, 1) # (N, 64, 32, 32)

        out = self.layer1(out) # (N, 64, 32, 32)
        out = self.layer2(out) # (N, 128, 16, 16)
        out = self.layer3(out) # (N, 256, 8, 8)
        out = self.layer4(out) # (N, 512, 4, 4)
        out = F.adaptive_avg_pool2d(out, 1)
        pre_out = out.view(out.size(0), -1)
        final = self.linear(pre_out)
        return final
    

def ResNet152Nemb(**kwargs):
    return ResNetNemb(Bottleneck, [3,8,36,3], **kwargs)

resnet152nemb = ResNet152Nemb