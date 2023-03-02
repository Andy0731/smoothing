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
from archs.normal_resnet import resnet300 as normal_resnet300
from archs.normal_resnet import resnet152wide2 as normal_resnet152wide2
from archs.normal_resnet_gelu import resnet152gelu as normal_resnet152_gelu
from archs.normal_resnet_nost import resnet152nost as normal_resnet152_nost
from archs.normal_resnet_avgn import resnet152avgn as normal_resnet152_avgn
from archs.vit import vit_b

import torchvision.models as torchvision_models

# resnet50 - the classic ResNet-50, sized for ImageNet
# cifar_resnet20 - a 20-layer residual network sized for CIFAR
# cifar_resnet110 - a 110-layer residual network sized for CIFAR
ARCHITECTURES = ["resnet50", "cifar_resnet20", "cifar_resnet110", 
                "normal_resnet18", "normal_resnet18wide", "normal_resnet34", "normal_resnet50", "normal_resnet101", "normal_resnet152",
                "normal_resnet300", "normal_resnet152wide2"]


def get_architecture(arch: str, dataset: str, avgn_loc: str = None, avgn_num: int = 1) -> torch.nn.Module:
    """ Return a neural network (with random weights)

    :param arch: the architecture - should be in the ARCHITECTURES list above
    :param dataset: the dataset - should be in the datasets.DATASETS list
    :return: a Pytorch module
    """
    print('arch:', arch, ' dataset: ', dataset)
    if arch == "resnet50" and dataset == "imagenet":
        model = resnet50()
        cudnn.benchmark = True
    elif arch == "cifar_resnet20":
        model = resnet_cifar(depth=20, num_classes=10 if 'cifar' in dataset else 1000)
    elif arch == "cifar_resnet110":
        model = resnet_cifar(depth=110, num_classes=10 if 'cifar' in dataset else 1000)
    elif arch == "cifar_resnet1199":
        model = resnet_cifar(depth=1199, num_classes=10 if 'cifar' in dataset else 1000, block_name='bottleneck')
    elif "normal_" in arch: # conv1 3x3
        if ('cifar' in dataset) or ('ti500k' in dataset):
            if 'avgn' in arch:
                model = arch + '(avgn_loc=avgn_loc, avgn_num=avgn_num)'
            else:
                model = arch + '()'
        elif dataset == 'imagenet22k':
            model = arch + '(num_classes=21841)'
        elif 'imagenet' in dataset:
            model = arch + '(num_classes=1000)'
        print('model: ', model)
        model = eval(model)
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
    return torch.nn.Sequential(normalize_layer, model)
