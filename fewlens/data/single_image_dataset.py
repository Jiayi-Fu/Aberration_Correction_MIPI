from os import path as osp
from torch.utils import data as data
from torchvision.transforms.functional import normalize

from fewlens.data.data_util import paths_from_lmdb
from fewlens.utils import FileClient, imfrombytes, img2tensor, rgb2ycbcr, scandir
from fewlens.utils.registry import DATASET_REGISTRY
import torch
import numpy as np
import cv2
@DATASET_REGISTRY.register()
class SingleImageDataset(data.Dataset):
    """Read only lq images in the test phase.

    Read LQ (Low Quality, e.g. LR (Low Resolution), blurry, noisy, etc).

    There are two modes:
    1. 'meta_info_file': Use meta information file to generate paths.
    2. 'folder': Scan folders to generate paths.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_lq (str): Data root path for lq.
            meta_info_file (str): Path for meta information file.
            io_backend (dict): IO backend type and other kwarg.
    """

    def __init__(self, opt):
        super(SingleImageDataset, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None
        self.lq_folder = opt['dataroot_lq']

        if self.io_backend_opt['type'] == 'lmdb':
            self.io_backend_opt['db_paths'] = [self.lq_folder]
            self.io_backend_opt['client_keys'] = ['lq']
            self.paths = paths_from_lmdb(self.lq_folder)
        elif 'meta_info_file' in self.opt:
            with open(self.opt['meta_info_file'], 'r') as fin:
                self.paths = [osp.join(self.lq_folder, line.rstrip().split(' ')[0]) for line in fin]
        else:
            self.paths = sorted(list(scandir(self.lq_folder, full_path=True)))

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        # load lq image
        lq_path = self.paths[index]
        img_bytes = self.file_client.get(lq_path, 'lq')
        img_lq = imfrombytes(img_bytes, float32=True)
        img_lq = cv2.cvtColor(img_lq, cv2.COLOR_BGR2RGB)



        [h, w, _] = img_lq.shape
        wsz = 32

        img_lq = img_lq[h%wsz:, w%wsz:,:]
        img_gt = img_gt[h%wsz:, w%wsz:,:]
        [h, w, _] = img_lq.shape
        ######################################################################################
        # calculate the fov information
        h_range = np.arange(0, h, 1)
        w_range = np.arange(0, w, 1)
        img_fld_w, img_fld_h = np.meshgrid(w_range, h_range)
        img_fld_h = ((img_fld_h - (h-1)/2) / ((h-1)/2)).astype(np.float32)
        img_fld_w = ((img_fld_w - (w-1)/2) / ((w-1)/2)).astype(np.float32)
        img_fld_h = np.expand_dims(img_fld_h, -1)
        img_fld_w = np.expand_dims(img_fld_w, -1)
        img_wz_fld = np.concatenate([img_lq, img_fld_h, img_fld_w], 2)


        # BGR to RGB, HWC to CHW, numpy to tensor
        img_lq_fld = img2tensor(img_wz_fld, bgr2rgb=False, float32=True)


        # normalize
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
        return {'lq': img_lq_fld, 'lq_path': lq_path}


    def __len__(self):
        return len(self.paths)
