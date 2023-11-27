import torch
from torchvision.models.resnet import resnet50
import torch.backends.cudnn as cudnn
from archs.cifar_resnet import resnet as resnet_cifar
from datasets import get_normalize_layer
from archs.normal_resnet import resnet18 as normal_resnet18
from archs.normal_resnet import resnet18wide as normal_resnet18wide
from archs.normal_resnet import resnet34 as normal_resnet34
from archs.normal_resnet import resnet50 as normal_resnet50
from archs.normal_resnet import resnet101 as normal_resnet101
from archs.normal_resnet import resnet152 as normal_resnet152
from archs.normal_resnet import resnet200 as normal_resnet200
from archs.normal_resnet import resnet300 as normal_resnet300
from archs.normal_resnet import resnet152wide2 as normal_resnet152wide2
from archs.normal_resnet_gelu import resnet152gelu as normal_resnet152_gelu
from archs.normal_resnet_nost import resnet152nost as normal_resnet152_nost
from archs.normal_resnet_avgn import resnet152avgn as normal_resnet152_avgn
from archs.normal_resnet_nconv import resnet152nconv as normal_resnet152_nconv
from archs.normal_resnet_nemb import resnet152nemb as normal_resnet152_nemb, TimestepEmbedSequential
from archs.normal_resnet_nemb_blk import resnetblkemd152 as normal_resnet152_nembblk
from archs.normal_resnet_in import resnet152in as normal_resnet152_in
from archs.normal_resnet_gn import resnet152gn as normal_resnet152_gn
from archs.normal_resnet_gn_efc import resnet152gnefc as normal_resnet152_gn_efc
from archs.normal_resnet_nt import resnet152nt as normal_resnet152_nt
from archs.vit import vit_b
from datasets import NormalizeLayer
from archs.torchvision_resnet_gn import resnet152tvgn as torchvision_resnet152gn

import torchvision.models as torchvision_models
import torch.nn as nn

# resnet50 - the classic ResNet-50, sized for ImageNet
# cifar_resnet20 - a 20-layer residual network sized for CIFAR
# cifar_resnet110 - a 110-layer residual network sized for CIFAR
ARCHITECTURES = ["resnet50", "cifar_resnet20", "cifar_resnet110", 
                "normal_resnet18", "normal_resnet18wide", "normal_resnet34", 
                "normal_resnet50", "normal_resnet101", "normal_resnet152",
                "normal_resnet200", "normal_resnet300", "normal_resnet152wide2"]

class ArgSequential(torch.nn.Sequential):
    """
    A sequential module that passes arg to the children that
    support it as an extra input.
    """
    def forward(self, x, **kwargs):
        for layer in self:
            if isinstance(layer, NormalizeLayer):
                x = layer(x)
            else:
                x = layer(x, **kwargs)
        return x


def get_architecture(arch: str, 
                     dataset: str,
                     class_num: int = None, 
                     avgn_loc: str = None, 
                     avgn_num: int = 1, 
                     nemb_layer: str = None, 
                     emb_scl=None, 
                     emb_dim=None, 
                     groups=None,
                     track_running_stats=True,
                     extra_fc_dim=None,
                     weights=None) -> torch.nn.Module:
    """ Return a neural network (with random weights)

    :param arch: the architecture - should be in the ARCHITECTURES list above
    :param dataset: the dataset - should be in the datasets.DATASETS list
    :return: a Pytorch module
    """
    print('arch:', arch, ' dataset: ', dataset)
    if arch == "resnet50" and dataset == "imagenet":
        model = resnet50()
        cudnn.benchmark = True
    elif arch == 'torchvision_resnet152':
        if weights == 'DEFAULT':
            model = torchvision_models.resnet152(weights=weights)
            model.fc = nn.Linear(2048, class_num)
        else:
            model = torchvision_models.resnet152(num_classes=class_num)
        cudnn.benchmark = True
    elif arch == 'torchvision_resnet50':
        if weights == 'DEFAULT':
            model = torchvision_models.resnet50(weights=weights)
            model.fc = nn.Linear(2048, class_num)
        else:
            model = torchvision_models.resnet50(num_classes=class_num)
        cudnn.benchmark = True
    elif arch == 'torchvision_resnet18':
        if weights == 'DEFAULT':
            model = torchvision_models.resnet18(weights=weights)
            model.fc = nn.Linear(512, class_num)
        else:
            model = torchvision_models.resnet18(num_classes=class_num)
        cudnn.benchmark = True
    elif arch == 'torchvision_resnet152gn':
        model = torchvision_resnet152gn(num_classes=class_num, gn_groups=groups)
        cudnn.benchmark = True
    elif arch == "cifar_resnet20":
        model = resnet_cifar(depth=20, num_classes=10 if 'cifar' in dataset else 1000)
    elif arch == "cifar_resnet110":
        model = resnet_cifar(depth=110, num_classes=10 if 'cifar' in dataset else 1000)
    elif arch == "cifar_resnet1199":
        model = resnet_cifar(depth=1199, num_classes=10 if 'cifar' in dataset else 1000, block_name='bottleneck')
    elif arch == "normal_resnet152_in":
        model = normal_resnet152_in(num_classes=class_num, track_running_stats=track_running_stats)
    elif arch == "normal_resnet152_nt":
        model = normal_resnet152_nt(num_classes=class_num, track_running_stats=track_running_stats)
    elif arch == "normal_resnet152_gn":
        model = normal_resnet152_gn(num_classes=class_num, groups=groups)
    elif arch == "normal_resnet152_gn_efc":
        model = normal_resnet152_gn_efc(num_classes=class_num, groups=groups, extra_fc_dim=extra_fc_dim)    
    elif "normal_" in arch: # conv1 3x3
        if ('cifar' in dataset) or ('ti500k' in dataset):
            if 'avgn' in arch:
                model = arch + '(avgn_loc=avgn_loc, avgn_num=avgn_num, num_classes=class_num)'
            elif 'nconv' in arch:
                model = arch + '(avgn_num=avgn_num, num_classes=class_num)'
            elif 'nemb' in arch:
                model = arch + '(nemb_layer=nemb_layer, emb_scl=emb_scl, emb_dim=emb_dim, num_classes=class_num)'
            else:
                # model = arch + '(num_classes=class_num, track_running_stats=track_running_stats)'
                model = arch + '(num_classes=class_num)'
        else:
            model = arch + '(num_classes=class_num)'
        print('model: ', model)
        model = eval(model)
        # print(model)
    elif arch == 'torchvision_resnet152': # conv1 7x7 with stride=2 and maxpooling before the 4 stages
        model = torchvision_models.resnet152(num_classes=10)
    elif 'vit' in arch:
        if 'cifar' in dataset:
            model = arch + '(num_classes=10)'
        elif dataset == 'imagenet32':
            model = arch + '(num_classes=1000)'
        elif dataset == 'imagenet22k':
            model = arch + '(num_classes=21841)'
        else:
            raise ValueError
        print('model: ', model)
        model = eval(model)
        print(model)
    else:
        raise ValueError
    normalize_layer = get_normalize_layer(dataset)

    if nemb_layer is not None:
        return TimestepEmbedSequential(normalize_layer, model)

    return ArgSequential(normalize_layer, model)
