from torchvision import transforms, datasets
from typing import *
import torch
import os
from torch.utils.data import Dataset
import pickle
import numpy as np
from PIL import Image

import moco.loader

# set this environment variable to the location of your imagenet directory if you want to read ImageNet data.
# make sure your val directory is preprocessed to look like the train directory, e.g. by running this script
# https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh
IMAGENET_LOC_ENV = "IMAGENET_DIR"

# list of all datasets
DATASETS = ["imagenet", "cifar10"]


def get_dataset(dataset: str, 
    split: str, 
    datapath: str = None, 
    dataaug: str = None, 
    noise_sd: float = 0.0) -> Dataset:

    """Return the dataset as a PyTorch Dataset object"""
    if dataset == "imagenet":
        return _imagenet(split, datapath, dataaug)
    elif dataset == "cifar10":
        return _cifar10(split, datapath, dataaug)
    elif dataset == 'imagenet32':
        return _imagenet32(split, datapath, dataaug, noise_sd)
    elif dataset == 'ti500k':
        return TiTop50KDataset(datapath)


def get_num_classes(dataset: str):
    """Return the number of classes in the dataset. """
    if "imagenet" in dataset:
        return 1000
    elif dataset == "cifar10":
        return 10
    elif dataset == "ti500k":
        return 10


def get_normalize_layer(dataset: str) -> torch.nn.Module:
    """Return the dataset's normalization layer"""
    if dataset == "imagenet":
        return NormalizeLayer(_IMAGENET_MEAN, _IMAGENET_STDDEV)
    elif dataset == "cifar10":
        return NormalizeLayer(_CIFAR10_MEAN, _CIFAR10_STDDEV)
    elif dataset == "imagenet32":
        return NormalizeLayer(_CIFAR10_MEAN, _CIFAR10_STDDEV)
    elif dataset == "ti500k":
        return NormalizeLayer(_CIFAR10_MEAN, _CIFAR10_STDDEV)


_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STDDEV = [0.229, 0.224, 0.225]

_CIFAR10_MEAN = [0.4914, 0.4822, 0.4465]
_CIFAR10_STDDEV = [0.2023, 0.1994, 0.2010]


def _cifar10(split: str, datapath: str = None, dataaug: str = None) -> Dataset:
    if split == "train":
        if dataaug == 'moco_v3_aug':
            img_transforms = transforms.Compose([
                transforms.RandomResizedCrop(32, scale=(0.2, 1.0)),
                transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomApply([transforms.GaussianBlur(kernel_size=(3, 7), sigma=(0.1, 2.0))], p=1.0),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ])
        elif dataaug == 'moco_mix_aug':
            img_transforms = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomApply([transforms.ColorJittekar(0.4, 0.4, 0.2, 0.1)], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomApply([transforms.GaussianBlur(kernel_size=(3, 7), sigma=(0.1, 2.0))], p=1.0),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ])            
        else:
            img_transforms = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor()
        ])
        return datasets.CIFAR10(datapath if datapath else "./dataset_cache", train=True, download=True, transform=img_transforms)
    elif split == "test":
        return datasets.CIFAR10(datapath if datapath else "./dataset_cache", train=False, download=True, transform=transforms.ToTensor())


def _imagenet(split: str, datapath: str = None, dataaug: str = None) -> Dataset:
    # if not IMAGENET_LOC_ENV in os.environ:
    #     raise RuntimeError("environment variable for ImageNet directory not set")
    if dataaug:
        print('Error! Custom data augmentation on ImageNet has not been implemented!')

    # dir = os.environ[IMAGENET_LOC_ENV]
    if split == "train":
        subdir = os.path.join(datapath, "train")
        transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])
    elif split == "test":
        subdir = os.path.join(datapath, "val")
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor()
        ])
    return datasets.ImageFolder(subdir, transform)


def _imagenet32(split: str, datapath: str = None, dataaug: str = None, noise_sd: float = 0.0) -> Dataset:
    if split == "train":
        if dataaug == 'moco_v3_aug':
            img_transforms = transforms.Compose([
                transforms.RandomResizedCrop(32),
                transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomApply([transforms.GaussianBlur(kernel_size=(3, 7), sigma=(0.1, 2.0))], p=1.0),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ])
        elif dataaug == 'moco_mix_aug':
            img_transforms = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomApply([transforms.ColorJittekar(0.4, 0.4, 0.2, 0.1)], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomApply([transforms.GaussianBlur(kernel_size=(3, 7), sigma=(0.1, 2.0))], p=1.0),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ])
        elif dataaug == 'moco_two_crops':
            normalize = transforms.Normalize(mean=_CIFAR10_MEAN, std=_CIFAR10_STDDEV)
            # follow BYOL's augmentation recipe: https://arxiv.org/abs/2006.07733
            augmentation1 = [
                transforms.RandomCrop(32, padding=4), 
                transforms.RandomApply([
                    transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)  # not strengthened
                ], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomApply([moco.loader.GaussianBlur([.1, 2.])], p=1.0),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                moco.loader.GaussianNoise(noise_sd),
                normalize
            ]

            augmentation2 = [
                transforms.RandomCrop(32, padding=4), 
                transforms.RandomApply([
                    transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)  # not strengthened
                ], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomApply([moco.loader.GaussianBlur([.1, 2.])], p=0.1),
                transforms.RandomApply([moco.loader.Solarize()], p=0.2),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                moco.loader.GaussianNoise(noise_sd),
                normalize
            ]            

            img_transforms = moco.loader.TwoCropsTransform(transforms.Compose(augmentation1), 
                transforms.Compose(augmentation2))

        elif dataaug == 'moco_two_noise':
            normalize = transforms.Normalize(mean=_CIFAR10_MEAN, std=_CIFAR10_STDDEV)
            # follow BYOL's augmentation recipe: https://arxiv.org/abs/2006.07733
            augmentation1 = [
                transforms.RandomCrop(32, padding=4), 
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                moco.loader.GaussianNoise(noise_sd),
                normalize
            ]

            augmentation2 = [
                transforms.RandomCrop(32, padding=4), 
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                moco.loader.GaussianNoise(noise_sd),
                normalize
            ]            

            img_transforms = moco.loader.TwoCropsTransform(transforms.Compose(augmentation1), 
                transforms.Compose(augmentation2))       

        elif dataaug == 'moco_two_only_noise':
            normalize = transforms.Normalize(mean=_CIFAR10_MEAN, std=_CIFAR10_STDDEV)
            # follow BYOL's augmentation recipe: https://arxiv.org/abs/2006.07733
            augmentation1 = [
                transforms.ToTensor(),
                moco.loader.GaussianNoise(noise_sd),
                normalize
            ]

            augmentation2 = [
                transforms.ToTensor(),
                moco.loader.GaussianNoise(noise_sd),
                normalize
            ]            

            img_transforms = moco.loader.TwoCropsTransform(transforms.Compose(augmentation1), 
                transforms.Compose(augmentation2))

        elif dataaug == 'moco_1crop2noise':
            normalize = transforms.Normalize(mean=_CIFAR10_MEAN, std=_CIFAR10_STDDEV)
            noise1 = transforms.Compose([
                moco.loader.GaussianNoise(noise_sd),
                normalize
            ])
            noise2 = transforms.Compose([
                moco.loader.GaussianNoise(noise_sd),
                normalize
            ])
            twonoise = moco.loader.TwoCropsTransform(noise1, noise2)

            img_transforms = transforms.Compose([
                transforms.RandomCrop(32, padding=4), 
                transforms.RandomApply([
                    transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)  # not strengthened
                ], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomApply([moco.loader.GaussianBlur([.1, 2.])], p=0.1),
                transforms.RandomApply([moco.loader.Solarize()], p=0.2),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                twonoise,
            ])

        else:
            img_transforms = transforms.Compose([
                transforms.RandomCrop(32, padding=4), 
                transforms.RandomHorizontalFlip(), 
                transforms.ToTensor()
            ])
        return ImageNetDS(datapath, 32, train=True, transform=img_transforms)
    elif split == "test":
        return ImageNetDS(datapath, 32, train=False, transform=transforms.ToTensor())


class NormalizeLayer(torch.nn.Module):
    """Standardize the channels of a batch of images by subtracting the dataset mean
      and dividing by the dataset standard deviation.

      In order to certify radii in original coordinates rather than standardized coordinates, we
      add the Gaussian noise _before_ standardizing, which is why we have standardization be the first
      layer of the classifier rather than as a part of preprocessing as is typical.
      """

    def __init__(self, means: List[float], sds: List[float]):
        """
        :param means: the channel means
        :param sds: the channel standard deviations
        """
        super(NormalizeLayer, self).__init__()
        self.means = torch.tensor(means).cuda()
        self.sds = torch.tensor(sds).cuda()

    def forward(self, input: torch.tensor):
        (batch_size, num_channels, height, width) = input.shape
        means = self.means.repeat((batch_size, height, width, 1)).permute(0, 3, 1, 2)
        sds = self.sds.repeat((batch_size, height, width, 1)).permute(0, 3, 1, 2)
        return (input - means) / sds


# from https://github.com/hendrycks/pre-training
class ImageNetDS(Dataset):
    """`Downsampled ImageNet <https://patrykchrabaszcz.github.io/Imagenet32/>`_ Datasets.
    Args:
        root (string): Root directory of dataset where directory
            ``ImagenetXX_train`` exists.
        img_size (int): Dimensions of the images: 64,32,16,8
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """
    train_list = [
        ['train_data_batch_1', ''],
        ['train_data_batch_2', ''],
        ['train_data_batch_3', ''],
        ['train_data_batch_4', ''],
        ['train_data_batch_5', ''],
        ['train_data_batch_6', ''],
        ['train_data_batch_7', ''],
        ['train_data_batch_8', ''],
        ['train_data_batch_9', ''],
        ['train_data_batch_10', '']
    ]

    test_list = [
        ['val_data', ''],
    ]

    def __init__(self, root, img_size, train=True, transform=None, target_transform=None):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set
        self.img_size = img_size

        # if not self._check_integrity():
        #    raise RuntimeError('Dataset not found or corrupted.') # TODO

        # now load the picked numpy arrays
        if self.train:
            self.train_data = []
            self.train_labels = []
            for fentry in self.train_list:
                f = fentry[0]
                file = os.path.join(self.root, f)
                with open(file, 'rb') as fo:
                    entry = pickle.load(fo)
                    self.train_data.append(entry['data'])
                    self.train_labels += [label - 1 for label in entry['labels']]
                    self.mean = entry['mean']

            self.train_data = np.concatenate(self.train_data)
            self.train_data = self.train_data.reshape((self.train_data.shape[0], 3, 32, 32))
            self.train_data = self.train_data.transpose((0, 2, 3, 1))  # convert to HWC
        else:
            f = self.test_list[0][0]
            file = os.path.join(self.root, f)
            fo = open(file, 'rb')
            entry = pickle.load(fo)
            self.test_data = entry['data']
            self.test_labels = [label - 1 for label in entry['labels']]
            fo.close()
            self.test_data = self.test_data.reshape((self.test_data.shape[0], 3, 32, 32))
            self.test_data = self.test_data.transpose((0, 2, 3, 1))  # convert to HWC

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.train:
            img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

    def _check_integrity(self):
        root = self.root
        for fentry in (self.train_list + self.test_list):
            filename, md5 = fentry[0], fentry[1]
            fpath = os.path.join(root, self.base_folder, filename)
            if not check_integrity(fpath, md5):
                return False
        return True


## To use this dataset, please contact the authors of https://arxiv.org/pdf/1905.13736.pdf
# to get access to this pickle file (ti_top_50000_pred_v3.1.pickle) containing the dataset.
class TiTop50KDataset(Dataset):
    """500K images closest to the CIFAR-10 dataset from 
        the 80 Millon Tiny Images Datasets"""
    def __init__(self, datapath):
        super(TiTop50KDataset, self).__init__()
        dataset_path = os.path.join(datapath, 'ti_500K_pseudo_labeled.pickle')

        self.dataset_dict = pickle.load(open(dataset_path,'rb'))
        #{'data', 'extrapolated_targets', 'ti_index', 
        # 'prediction_model', 'prediction_model_epoch'}
        
        self.length = len(self.dataset_dict['data'])
        self.transforms = transforms.Compose([
                    transforms.Resize((32,32)),
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor()
                ])

    def __getitem__(self, index):
        img = self.dataset_dict['data'][index]
        target = self.dataset_dict['extrapolated_targets'][index]
        
        img = Image.fromarray(img)
        img = self.transforms(img)

        return img, target
                
    def __len__(self):
        return self.length