import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torch.distributed as dist

import numpy as np
from PIL import Image, ImageFilter
import time
import csv
import sys
import os
import base64
import math
import os.path as op
import random
import io

csv.field_size_limit(sys.maxsize)

def img_from_base64(imagestring, color=True):
    img_str = base64.b64decode(imagestring)
    try:
        if color:
            r = Image.open(io.BytesIO(img_str)).convert('RGB')
            return r
        else:
            r = Image.open(io.BytesIO(img_str)).convert('L')
            return r
    except:
        return None

def generate_lineidx(filein, idxout):
    assert not os.path.isfile(idxout)
    with open(filein, 'r') as tsvin, open(idxout, 'w') as tsvout:
        fsize = os.fstat(tsvin.fileno()).st_size
        fpos = 0
        while fpos != fsize:
            tsvout.write(str(fpos)+"\n")
            tsvin.readline()
            fpos = tsvin.tell()

class TSVFile(object):
    def __init__(self, tsv_file):
        self.tsv_file = tsv_file
        self.lineidx = op.splitext(tsv_file)[0] + '.lineidx'
        self._fp = None
        self._lineidx = None

    def num_rows(self):
        self._ensure_lineidx_loaded()
        return len(self._lineidx)

    def seek(self, idx):
        self._ensure_tsv_opened()
        self._ensure_lineidx_loaded()
        pos = self._lineidx[idx]
        self._fp.seek(pos)
        return [s.strip() for s in self._fp.readline().split('\t')]

    def seek_list(self, idxs, q):
        assert isinstance(idxs, list)
        self._ensure_tsv_opened()
        self._ensure_lineidx_loaded()
        for idx in idxs:
            pos = self._lineidx[idx]
            self._fp.seek(pos)
            q.put([s.strip() for s in self._fp.readline().split('\t')])

    def close(self):
        if self._fp is not None:
            self._fp.close()
            self._fp = None

    def _ensure_lineidx_loaded(self):
    #    if not op.isfile(self.lineidx) and not op.islink(self.lineidx):
    #        generate_lineidx(self.tsv_file, self.lineidx)
        if self._lineidx is None:
            with open(self.lineidx, 'r') as fp:
                self._lineidx = [int(i.strip()) for i in fp.readlines()]

    def _ensure_tsv_opened(self):
        if self._fp is None:
            self._fp = open(self.tsv_file, 'r')

class TSVInstance(Dataset):
    def __init__(self, tsv_file, transform=None, two_crop=False):
        self.tsv = TSVFile(tsv_file + '.tsv')
        self.transform = transform
        self.two_crop = two_crop

    def __getitem__(self, index):
        row = self.tsv.seek(index)
        image = img_from_base64(row[-1])

        if self.transform is not None:
            img = self.transform(image)
        else:
            img = image

        if self.two_crop:
            img2 = self.transform(image)
            img = torch.cat([img, img2], dim=0)

        label = int(row[1])
        label = torch.from_numpy(np.array(label, dtype=np.int))

        return img, label

    def __len__(self):
        return self.tsv.num_rows()

class TSVInstancePreload(Dataset):
    def __init__(self, tsv_file, transform=None, two_crop=True):
        self.tsv = TSVFile(tsv_file + '.tsv')
        self.transform = transform
        self.two_crop = two_crop

        self.ndata = self.tsv.num_rows()
        self.image = [None for _ in range(self.ndata)]

        start_time = time.time()
        for i in range(self.ndata):
            if i % (self.ndata//3) == 0:
                print('rank: {} cached {}/{} takes {:.2f}s per block'.format(dist.get_rank(), i, self.ndata, time.time()-start_time))
                start_time = time.time()
            if i % dist.get_world_size() == dist.get_rank():
                path, target, binary = self.tsv.seek(i)
                self.image[i] = (binary, path, target)
            else:
                self.image[i] = (None, None, None)
        
    def __getitem__(self, index):
        image = img_from_base64(self.image[index][0])

        if self.transform is not None:
            img = self.transform(image)
        else:
            img = image

        if self.two_crop:
            img2 = self.transform(image)
            img = torch.cat([img, img2], dim=0)

        label = int(self.image[index][2])
        label = torch.from_numpy(np.array(label, dtype=np.int))

        return img, label

    def __len__(self):
        return self.ndata

