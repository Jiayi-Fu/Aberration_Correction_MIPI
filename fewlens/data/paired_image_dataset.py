from torch.utils import data as data
from torchvision.transforms.functional import normalize

from fewlens.data.data_util import (paired_paths_from_folder,
                                    paired_paths_from_lmdb,
                                    paired_paths_from_meta_info_file)
from fewlens.data.transforms import augment, paired_random_crop, random_augmentation
from fewlens.utils import FileClient, imfrombytes, img2tensor, padding, padding_DP, imfrombytesDP
import os
from fewlens.utils.registry import DATASET_REGISTRY
import numpy as np
import torch
import cv2
import scipy.io
@DATASET_REGISTRY.register()
class Dataset_PairedImage(data.Dataset):
    """Paired image dataset for image restoration.

    Read LQ (Low Quality, e.g. LR (Low Resolution), blurry, noisy, etc) and
    GT image pairs.

    There are three modes:
    1. 'lmdb': Use lmdb files.
        If opt['io_backend'] == lmdb.
    2. 'meta_info_file': Use meta information file to generate paths.
        If opt['io_backend'] != lmdb and opt['meta_info_file'] is not None.
    3. 'folder': Scan folders to generate paths.
        The rest.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            dataroot_lq (str): Data root path for lq.
            meta_info_file (str): Path for meta information file.
            io_backend (dict): IO backend type and other kwarg.
            filename_tmpl (str): Template for each filename. Note that the
                template excludes the file extension. Default: '{}'.
            gt_size (int): Cropped patched size for gt patches.
            geometric_augs (bool): Use geometric augmentations.

            scale (bool): Scale, which will be added automatically.
            phase (str): 'train' or 'val'.
    """

    def __init__(self, opt):
        super(Dataset_PairedImage, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None

        self.PE = opt['PE'] if 'PE' in opt else False
        self.return_psf = True if 'return_psf' in opt else False
        self.gt_folder, self.lq_folder = opt['dataroot_gt'], opt['dataroot_lq']

        if 'filename_tmpl' in opt:
            self.filename_tmpl = opt['filename_tmpl']
        else:
            self.filename_tmpl = '{}'

        if self.io_backend_opt['type'] == 'lmdb':
            self.io_backend_opt['db_paths'] = [self.lq_folder, self.gt_folder]
            self.io_backend_opt['client_keys'] = ['lq', 'gt']
            self.paths = paired_paths_from_lmdb(
                [self.lq_folder, self.gt_folder], ['lq', 'gt'])
        elif 'meta_info_file' in self.opt and self.opt[
                'meta_info_file'] is not None:
            self.paths = paired_paths_from_meta_info_file(
                [self.lq_folder, self.gt_folder], ['lq', 'gt'],
                self.opt['meta_info_file'], self.filename_tmpl)
        else:
            self.paths = paired_paths_from_folder(
                [self.lq_folder, self.gt_folder], ['lq', 'gt'],
                self.filename_tmpl)


    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt['scale']
        index = index % len(self.paths)
        # Load gt and lq images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.


        gt_path = self.paths[index]['gt_path']
        img_bytes = self.file_client.get(gt_path, 'gt')
        try:
            img_gt = imfrombytes(img_bytes, float32=True)
            img_gt = cv2.cvtColor(img_gt, cv2.COLOR_BGR2RGB)
        except:
            raise Exception("gt path {} not working".format(gt_path))

        lq_path = self.paths[index]['lq_path']
        img_bytes = self.file_client.get(lq_path, 'lq')
        try:
            img_lq = imfrombytes(img_bytes, float32=True)
            # img_lq = cv2.normalize(img_lq, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

            img_lq = cv2.cvtColor(img_lq, cv2.COLOR_BGR2RGB)
        except:
            raise Exception("lq path {} not working".format(lq_path))
        
        if self.opt['phase'] != 'train':
            h,w,c = img_lq.shape
            # if h * w < 2048 * 2048:
            #     img_lq = cv2.resize(img_lq, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
            #     img_gt = cv2.resize(img_gt, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
            # else:
            #     img_lq = cv2.resize(img_lq, (0, 0), fx=0.25, fy=0.25, interpolation=cv2.INTER_LINEAR)
            #     img_gt = cv2.resize(img_gt, (0, 0), fx=0.25, fy=0.25, interpolation=cv2.INTER_LINEAR)
                
        

        wsz = 64
        [h, w, _] = img_lq.shape
        img_lq = img_lq[h%wsz:, w%wsz:,:]
        img_gt = img_gt[h%wsz:, w%wsz:,:]


        [h, w, _] = img_lq.shape

        ######################################################################################
        # calculate the fov information
        
        h_range = np.arange(0, h, 1)
        w_range = np.arange(0, w, 1)
        img_fld_w, img_fld_h = np.meshgrid(w_range, h_range)
        
        if self.PE:
            img_fld_h = img_fld_h*3072/h
            img_fld_w = img_fld_w*4096/w
            img_fld_h = img_fld_h//8
            img_fld_w = img_fld_w//8
            # print(img_fld_h,img_fld_w)
        else:
            img_fld_h = ((img_fld_h - (h-1)/2) / ((h-1)/2)).astype(np.float32)
            img_fld_w = ((img_fld_w - (w-1)/2) / ((w-1)/2)).astype(np.float32)
        
        
        
        img_fld_h = np.expand_dims(img_fld_h, -1)
        img_fld_w = np.expand_dims(img_fld_w, -1)
        img_wz_fld = np.concatenate([img_lq, img_fld_h, img_fld_w], 2)


        # augmentation for training
        if self.opt['phase'] == 'train':
            gt_size = self.opt['gt_size']
            
            # # padding
            # img_gt, img_lq = padding(img_gt, img_lq, gt_size)

            # # random crop
            # img_gt, img_lq = paired_random_crop(img_gt, img_lq, gt_size, scale,
            #                                     gt_path)

            if gt_size is not None:
                half_patch_size = gt_size // 2
                (ht, wt) = (np.random.randint(half_patch_size, h - half_patch_size), \
                    np.random.randint(half_patch_size, w - half_patch_size))

                img_lq = img_wz_fld[ht - half_patch_size:ht + half_patch_size, wt - half_patch_size:wt + half_patch_size, :]
                img_gt = img_gt[ht - half_patch_size:ht + half_patch_size, wt - half_patch_size:wt + half_patch_size, :]
            else:
                raise ValueError(f'gt_size is None')

            # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt, img_lq = img2tensor([img_gt, img_wz_fld],
                                        bgr2rgb=False,
                                        float32=True)

        # normalize
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)
            

        if self.return_psf:
            psf = scipy.io.loadmat('datasets/vivo/data20241028/39525_pLensFittingPSF12x16_Obj200mm.mat')['PSF']
            psf_h,psf_w,_,_,_ = psf.shape
            psf = torch.from_numpy(psf).float().reshape(psf_h,psf_w,-1)
            psf = psf.permute(2,0,1)
            return {
                'lq': img_lq,
                'gt': img_gt,
                'lq_path': lq_path,
                'gt_path': gt_path,
                'psf_code':psf
            }
            
        return {
            'lq': img_lq,
            'gt': img_gt,
            'lq_path': lq_path,
            'gt_path': gt_path,
        }

    def __len__(self):
        return len(self.paths)
