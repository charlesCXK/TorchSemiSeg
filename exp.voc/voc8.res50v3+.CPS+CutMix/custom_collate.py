import io
import os
import math
import zipfile
import threading
import itertools
from collections import namedtuple
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, Sampler
from torch.utils.data.dataloader import default_collate # default_collate

class SegCollate(object):
    def __init__(self, batch_aug_fn=None):
        self.batch_aug_fn = batch_aug_fn

    def __call__(self, batch):
        if self.batch_aug_fn is not None:
            batch = self.batch_aug_fn(batch)
        return default_collate(batch)