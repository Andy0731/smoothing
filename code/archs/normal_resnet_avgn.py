import torch
import torch.nn as nn
import torch.nn.functional as F
from .normal_resnet import ResNet, Bottleneck


class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, avgn_num: int = 1, output_dim: int = None):
        super().__init__()
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads
        self.avgn_num = avgn_num

    def forward(self, x): # x: bn, c -> b, c
        c = x.shape[-1]
        x = x.view(-1, self.avgn_num, c) # b, n, c
        x = x.permute(1, 0, 2) # n, b, c
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (1+n), b, c

        x, _ = F.multi_head_attention_forward(
            query=x[:1], key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )
        del c
        del _
        return x.squeeze(0) # b, c


class ResNetAvgn(ResNet):
    def __init__(self, block, num_blocks, num_classes=10, feat_scale=1, wm=1, avgn_loc=None, avgn_num=1):
        super().__init__(block, num_blocks, num_classes=10, feat_scale=1, wm=1)
        self.avgn_loc = avgn_loc
        self.avgn_num = avgn_num
        if avgn_loc == 'atp_h16_fc':
            self.attpool = AttentionPool2d(4, 2048, 16, avgn_num, num_classes)
        elif avgn_loc == 'atp_h8_fc':
            self.attpool = AttentionPool2d(4, 2048, 8, avgn_num, num_classes)
        elif avgn_loc == 'atp_h16_pj':
            self.attpool = AttentionPool2d(4, 2048, 16, avgn_num)
        elif avgn_loc == 'atp_h8_pj':
            self.attpool = AttentionPool2d(4, 2048, 8, avgn_num)

    def forward(self, x, with_latent=False, fake_relu=False, no_relu=False):
        assert (not no_relu),  \
            "no_relu not yet supported for this architecture"
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out, fake_relu=fake_relu)
        out = F.avg_pool2d(out, 4)
        pre_out = out.view(out.size(0), -1) # bx, c
        if self.avgn_loc in ['atp_h16_fc', 'atp_h8_fc']: # bx, c(2048)
            final = self.attpool(pre_out) # b, c(10)
            return final
        if self.avgn_loc in ['atp_h16_pj', 'atp_h8_pj']: # bx, c(2048)
            pre_out = self.attpool(pre_out) # b, c(2048)
        elif self.avgn_loc == 'before_fc': # bx,c(2048)
            c = pre_out.shape[-1]
            pre_out = pre_out.view(-1, self.avgn_num, c) # b,x,c(2048)
            pre_out = pre_out.mean(dim=1) # b,c(2048)
            del c
        final = self.linear(pre_out) # b, c(10)
        if self.avgn_loc == 'after_fc': # bx,c(10)
            c = pre_out.shape[-1]
            final = final.view(-1, self.avgn_num, c) # b,x,c(10)
            final = final.mean(dim=1) # b,c(10) 
            del c           
        if with_latent:
            return final, pre_out
        return final


def ResNet152Avgn(**kwargs):
    return ResNetAvgn(Bottleneck, [3,8,36,3], **kwargs)  


resnet152avgn = ResNet152Avgn

