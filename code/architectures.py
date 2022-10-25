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
# resnet50 - the classic ResNet-50, sized for ImageNet
# cifar_resnet20 - a 20-layer residual network sized for CIFAR
# cifar_resnet110 - a 110-layer residual network sized for CIFAR
ARCHITECTURES = ["resnet50", "cifar_resnet20", "cifar_resnet110", 
                "normal_resnet18", "normal_resnet18wide", "normal_resnet34", "normal_resnet50", "normal_resnet101", "normal_resnet152"]


def get_architecture(arch: str, dataset: str) -> torch.nn.Module:
    """ Return a neural network (with random weights)

    :param arch: the architecture - should be in the ARCHITECTURES list above
    :param dataset: the dataset - should be in the datasets.DATASETS list
    :return: a Pytorch module
    """
    if arch == "resnet50" and dataset == "imagenet":
        model = torch.nn.DataParallel(resnet50(pretrained=False))
        cudnn.benchmark = True
    elif arch == "cifar_resnet20":
        model = resnet_cifar(depth=20, num_classes=10)
    elif arch == "cifar_resnet110":
        model = resnet_cifar(depth=110, num_classes=10)
    elif arch == "cifar_resnet1199":
        model = resnet_cifar(depth=1199, num_classes=10, block_name='bottleneck')
    elif "normal_" in arch:
        model = eval(arch + '()')
        print('model: ', model)
    normalize_layer = get_normalize_layer(dataset)
    return torch.nn.Sequential(normalize_layer, model)
