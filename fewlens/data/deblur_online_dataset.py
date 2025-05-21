from torch.utils import data as data
from torchvision.transforms.functional import normalize

from fewlens.data.data_util import (paths_from_folder,
                                    paired_paths_from_lmdb,
                                    paired_paths_from_meta_info_file)
from fewlens.data.transforms import augment, paired_random_crop, random_augmentation
from fewlens.utils import FileClient, imfrombytes, img2tensor, padding, padding_DP, imfrombytesDP
import os
from fewlens.utils.registry import DATASET_REGISTRY
import numpy as np
import torch
import random
from fewlens.utils.isp import addnoise,mosaic
import cv2
import scipy.io

ccm = np.array([[ 1.93994141, -0.73925781, -0.20068359],
                [-0.28857422,  1.59741211, -0.30883789],
                [-0.0078125 , -0.45654297,  1.46435547]])

wb_arr = np.array([1.8910, 1, 1.8031,
                1.8031, 1, 1.7400,
                2.0156, 1, 1.7308,
                1.7436, 1, 1.9560]).reshape(4, 3)
mean_values = np.array([2.43945312, 1, 1.59440104])
std_devs = np.array([0.09231486, 0, 0.04701403])
def apply_ccm(img, ccm, inverse=False):
    """Applies a color correction matrix."""
    if inverse:
        ccm = np.linalg.inv(ccm)

    # Reshape img for matrix multiplication
    img_reshaped = img.reshape(-1, 3).T  # 转换为 3 x N 形状
    img_out = ccm @ img_reshaped          # 应用色彩校正矩阵
    img_out = img_out.T.reshape(img.shape)  # 恢复原始形状

    return img_out

def apply_wb(img, wb, inverse=False):
    """Applies white balance to the image."""
    if inverse:
        wb = 1.0 / wb

    img_out = np.stack((
        img[:, :, 0] * wb[0],
        img[:, :, 1] * wb[1],
        img[:, :, 2] * wb[2]
    ), axis=-1)

    return img_out

@DATASET_REGISTRY.register()
class BlurOnlineDataset(data.Dataset):
    """Paired image dataset for image restoration.

    Read LQ (Low Quality, e.g. LR (Low Resolution), blurry, noisy, etc) and
    GT image pairs.
    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            io_backend (dict): IO backend type and other kwarg.
            filename_tmpl (str): Template for each filename. Note that the
                template excludes the file extension. Default: '{}'.
            gt_size (int): Cropped patched size for gt patches.
            geometric_augs (bool): Use geometric augmentations.

            scale (bool): Scale, which will be added automatically.
            phase (str): 'train' or 'val'.
    """

    def __init__(self, opt):
        super(BlurOnlineDataset, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None
        self.sigma = opt.get('sigma', 3)
        self.patch_size = opt['patch_size']
        self.gt_folder = opt['dataroot_gt']
        self.noise_folder = opt['dataroot_noise']
        self.psf_folder = opt['dataroot_psf']
        self.psf_size = opt['psf_size']
        
        # self.psf_bases = "psf_components.npy"
        # self.psf_bases = np.load(self.psf_bases)
        # self.psf_bases = torch.from_numpy(self.psf_bases).cuda().reshape(12,16,5,21,21)
        
        
        
        self.psf_paths = []
        if os.path.isdir(self.psf_folder):
            for file in os.listdir(self.psf_folder):
                self.psf_paths.append(os.path.join(self.psf_folder, file))
        else:
            self.psf_paths.append(self.psf_folder)
            
        self.h_orgin,self.w_orgin = opt['img_orgin'][:]
        self.noises = []
        for file in os.listdir(self.noise_folder):
            noise_path = os.path.join(self.noise_folder, file)
            noise = scipy.io.loadmat(noise_path)
            self.noises.append(noise)


        self.paths = paths_from_folder(self.gt_folder)


    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop('type'), **self.io_backend_opt)

        index = index % len(self.paths)
        # Load gt and lq images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.


        gt_path = self.paths[index]
        img_bytes = self.file_client.get(gt_path, 'gt')
        try:
            img_gt = imfrombytes(img_bytes, float32=True)
            img_gt = cv2.cvtColor(img_gt, cv2.COLOR_BGR2RGB)
        except:
            raise Exception("gt path {} not working".format(gt_path))


        psf_index_h,psf_index_w = gt_path.split('/')[-1].split('_')[1:3]
        psf_index_h,psf_index_w = eval(psf_index_h),eval(psf_index_w.split('.')[0])
        psf_path = self.psf_paths[np.random.randint(0, len(self.psf_paths))]
        psf = scipy.io.loadmat(psf_path)['PSF']

        # Convolution for each color channel
        conved_img_pad_patch_r = cv2.filter2D(img_gt[:, :, 0], -1, psf[psf_index_h, psf_index_w, :, :, 0], borderType=cv2.BORDER_REPLICATE)
        conved_img_pad_patch_g = cv2.filter2D(img_gt[:, :, 1], -1, psf[psf_index_h, psf_index_w, :, :, 1], borderType=cv2.BORDER_REPLICATE)
        conved_img_pad_patch_b = cv2.filter2D(img_gt[:, :, 2], -1, psf[psf_index_h, psf_index_w, :, :, 2], borderType=cv2.BORDER_REPLICATE)

        conved_img_pad_patch = np.stack((conved_img_pad_patch_r, conved_img_pad_patch_g, conved_img_pad_patch_b), axis=-1)

        img_lq = conved_img_pad_patch[self.psf_size //2: self.psf_size//2 + self.patch_size, self.psf_size//2: self.psf_size//2 + self.patch_size, :]
        img_lq = img_lq.clip(0,1)
        # add noise
        if self.opt['noise']:
            img_lq = addnoise(img_lq, self.noises[np.random.randint(0, len(self.noises))])

        if self.sigma:
            noise = np.random.normal(loc=0, scale=self.sigma/255.0, size=img_lq.shape)
            img_lq = img_lq + noise
            img_lq = np.clip(img_lq, 0.0, 1.0)

        img_gt = img_gt[self.psf_size //2: self.psf_size//2+ self.patch_size, self.psf_size//2: self.psf_size//2+ self.patch_size, :]

        h_range = np.arange(psf_index_h*self.patch_size, (psf_index_h + 1) * self.patch_size, 1)
        w_range = np.arange(psf_index_w*self.patch_size, (psf_index_w + 1) * self.patch_size, 1)
        img_fld_w, img_fld_h = np.meshgrid(w_range, h_range)
        

        img_fld_h = ((img_fld_h - (self.h_orgin-1)/2) / ((self.h_orgin-1)/2)).astype(np.float32)
        img_fld_w = ((img_fld_w - (self.w_orgin-1)/2) / ((self.w_orgin-1)/2)).astype(np.float32)

        img_fld_h = np.expand_dims(img_fld_h, -1)
        img_fld_w = np.expand_dims(img_fld_w, -1)
        img_wz_fld = np.concatenate([img_lq, img_fld_h, img_fld_w], 2)

        img_gt, img_wz_fld = augment([img_gt, img_wz_fld], self.opt['use_hflip'], self.opt['use_rot'])

        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt, img_lq = img2tensor([img_gt, img_wz_fld],
                                        bgr2rgb=False,
                                        float32=True)
        
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)
            
        return {
            'lq': img_lq,
            'gt': img_gt,
            'gt_path': gt_path,
            # "psf": psf,
            # "psf_bases": self.psf_bases[psf_index_h, psf_index_w]
        }
    def __len__(self):
        return len(self.paths)

@DATASET_REGISTRY.register()
class BlurOnlineDatasetV2(data.Dataset):
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
        super(BlurOnlineDatasetV2, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None
        self.sigma = opt['sigma'] if 'sigma' in opt else 3

        self.patch_size = opt['patch_size']

        self.gt_folder = opt['dataroot_gt']
        self.psf_folder = opt['dataroot_psf']
        self.noise_folder = opt['dataroot_noise'] if 'dataroot_noise' in opt else True

        self.psf_folder = opt['dataroot_psf']

        self.psf_size = opt['psf_size']
        
        self.psf_paths = []
        for file in os.listdir(self.psf_folder):
            self.psf_paths.append(os.path.join(self.psf_folder, file))

        self.h_orgin,self.w_orgin = opt['img_orgin'][:]
        self.noises = []
        for file in os.listdir(self.noise_folder):
            noise_path = os.path.join(self.noise_folder, file)
            noise = scipy.io.loadmat(noise_path)
            self.noises.append(noise)

        if 'filename_tmpl' in opt:
            self.filename_tmpl = opt['filename_tmpl']
        else:
            self.filename_tmpl = '{}'
            
        self.paths = paths_from_folder(self.gt_folder)

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop('type'), **self.io_backend_opt)

        index = index % len(self.paths)
        # Load gt and lq images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.

        gt_path = self.paths[index]
        img_bytes = self.file_client.get(gt_path, 'gt')
        try:
            img_gt = imfrombytes(img_bytes, float32=True)
            img_gt = cv2.cvtColor(img_gt, cv2.COLOR_BGR2RGB)
        except:
            raise Exception("gt path {} not working".format(gt_path))

        # psf_index_h,psf_index_w = gt_path.split('/')[-1].split('_')[1:3]
        # psf_index_h,psf_index_w = eval(psf_index_h),eval(psf_index_w.split('.')[0])
        psf_index_h = np.random.randint(0, 11)
        psf_index_w = np.random.randint(0, 15)
        psf_path = self.psf_paths[np.random.randint(0, len(self.psf_paths))]
        psf = scipy.io.loadmat(psf_path)['PSF']
        img_lq = np.zeros((self.patch_size,self.patch_size, 3))

        for patch_index_h in range(2):
            for patch_index_w in range(2):
                img_pad_patch = img_gt[patch_index_h * self.patch_size//2 : (patch_index_h + 1) * self.patch_size//2 + self.psf_size,
                                        patch_index_w * self.patch_size//2 : (patch_index_w + 1) * self.patch_size//2 + self.psf_size, :]
                psf_r = psf[psf_index_h+patch_index_h, psf_index_w+patch_index_w, :, :, 0]  # 红色通道的PSF
                psf_g = psf[psf_index_h+patch_index_h, psf_index_w+patch_index_w, :, :, 1]  # 绿色通道的PSF
                psf_b = psf[psf_index_h+patch_index_h, psf_index_w+patch_index_w, :, :, 2]  # 蓝色通道的PSF

                # 对每个通道进行卷积
                conved_img_patch_r = cv2.filter2D(img_pad_patch[:, :, 0], -1, psf_r, borderType=cv2.BORDER_REPLICATE)
                conved_img_patch_g = cv2.filter2D(img_pad_patch[:, :, 1], -1, psf_g, borderType=cv2.BORDER_REPLICATE)
                conved_img_patch_b = cv2.filter2D(img_pad_patch[:, :, 2], -1, psf_b, borderType=cv2.BORDER_REPLICATE)
                # conved_img_patch_r = img_pad_patch[:, :, 0]
                # conved_img_patch_g = img_pad_patch[:, :, 1]
                # conved_img_patch_b = img_pad_patch[:, :, 2]
                # 合并RGB通道
                conved_img_patch = np.stack((conved_img_patch_r, conved_img_patch_g, conved_img_patch_b), axis=-1)
                patch_size = self.patch_size
                psf_size = self.psf_size

                try:
                    img_lq[(patch_index_h) * self.patch_size//2 : (patch_index_h+1) * self.patch_size//2, 
                            (patch_index_w) * self.patch_size//2 : (patch_index_w+1) * self.patch_size//2, :] = conved_img_patch[self.psf_size//2:self.psf_size//2 + self.patch_size//2, \
                                self.psf_size//2:self.psf_size//2 + self.patch_size//2, :]
                except:
                    print((patch_index_h) * patch_size//2 , (patch_index_h+1) * patch_size//2, (patch_index_w) * patch_size//2 , (patch_index_w+1) * patch_size//2)
                    print(psf_size//2,psf_size//2 + patch_size//2, psf_size//2,psf_size//2 + patch_size//2)
                    print(gt_path)
        img_lq = img_lq.clip(0,1)
        # add noise
        if self.opt['noise']:
            img_lq = addnoise(img_lq, self.noises[np.random.randint(0, len(self.noises))])

        # mosic demosic
        # Mosaicing
        # img_mosaiced = mosaic(img_lq, 'rggb')

        # img_mosaiced = img_mosaiced*255.0
        # img_mosaiced = np.clip(img_mosaiced, 0, 255).astype(np.uint8)

        # img_mosaiced = img_mosaiced.reshape(self.patch_size // 2, self.patch_size // 2, 2, 2).transpose(0, 2, 1, 3).reshape(self.patch_size, self.patch_size)
        # img_lq = cv2.cvtColor(img_mosaiced, cv2.COLOR_BayerRGGB2RGB_EA)/255.0


        if self.sigma:
            noise = np.random.normal(loc=0, scale=self.sigma/255.0, size=img_lq.shape)
            img_lq = img_lq + noise
            img_lq = np.clip(img_lq, 0.0, 1.0)

        img_gt = img_gt[self.psf_size //2: self.psf_size//2+ self.patch_size, self.psf_size//2: self.psf_size//2+ self.patch_size, :]


        ######################################################################################
        # h_orgin,w_orgin = self.h_orgin,self.w_orgin #3072,4096

        h_range = np.arange(psf_index_h*self.patch_size, (psf_index_h + 1) * self.patch_size, 1)
        w_range = np.arange(psf_index_w*self.patch_size, (psf_index_w + 1) * self.patch_size, 1)
        img_fld_w, img_fld_h = np.meshgrid(w_range, h_range)
        

        img_fld_h = np.expand_dims(img_fld_h, -1)
        img_fld_w = np.expand_dims(img_fld_w, -1)
        img_wz_fld = np.concatenate([img_lq, img_fld_h, img_fld_w], 2)


        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt, img_lq = img2tensor([img_gt, img_wz_fld],
                                        bgr2rgb=False,
                                        float32=True)
        

        # psf_code = np.load('psf_code.npy').reshape(12,16,52)
        # psf_patch = psf_code[psf_index_h, psf_index_w]  #52
        # psf_patch = torch.from_numpy(psf_patch).float()
        # print(psf_patch.shape)

        # normalize
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)

        return {
            'lq': img_lq,
            'gt': img_gt,
            'gt_path': gt_path,
            # 'psf_code':psf_patch
        }

    def __len__(self):
        return len(self.paths)



@DATASET_REGISTRY.register()
class BlurOnlineDatasetV3(data.Dataset):
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
        super(BlurOnlineDatasetV3, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None
        self.sigma = opt['sigma'] if 'sigma' in opt else 5
        self.PE = opt['PE'] if 'PE' in opt else False
        
        self.patch_size = opt['patch_size']

        self.gt_folder = opt['dataroot_gt']
        self.psf_folder = opt['dataroot_psf']
        self.noise_folder = opt['dataroot_noise'] if 'dataroot_noise' in opt else True

        self.psf_folder = opt['dataroot_psf']

        self.psf_size = opt['psf_size']
        
        self.psf_paths = []
        for file in os.listdir(self.psf_folder):
            self.psf_paths.append(os.path.join(self.psf_folder, file))

        self.h_orgin,self.w_orgin = opt['img_orgin'][:]
        self.noises = []
        for file in os.listdir(self.noise_folder):
            noise_path = os.path.join(self.noise_folder, file)
            noise = scipy.io.loadmat(noise_path)
            self.noises.append(noise)

        if 'filename_tmpl' in opt:
            self.filename_tmpl = opt['filename_tmpl']
        else:
            self.filename_tmpl = '{}'
            
        self.paths = paths_from_folder(self.gt_folder)

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop('type'), **self.io_backend_opt)

        index = index % len(self.paths)
        # Load gt and lq images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.

        gt_path = self.paths[index]
        img_bytes = self.file_client.get(gt_path, 'gt')
        try:
            img_gt = imfrombytes(img_bytes, float32=True)
            img_gt = cv2.cvtColor(img_gt, cv2.COLOR_BGR2RGB)
        except:
            raise Exception("gt path {} not working".format(gt_path))

        # psf_index_h,psf_index_w = gt_path.split('/')[-1].split('_')[1:3]
        # psf_index_h,psf_index_w = eval(psf_index_h),eval(psf_index_w.split('.')[0])
        psf_index_h = np.random.randint(0, 11)
        psf_index_w = np.random.randint(0, 15)
        psf_path = self.psf_paths[np.random.randint(0, len(self.psf_paths))]
        psf = scipy.io.loadmat(psf_path)['PSF']
        img_lq = np.zeros((self.patch_size,self.patch_size, 3))

       # Convolution for each color channel
        conved_img_pad_patch_r = cv2.filter2D(img_gt[:, :, 0], -1, psf[psf_index_h, psf_index_w, :, :, 0], borderType=cv2.BORDER_REPLICATE)
        conved_img_pad_patch_g = cv2.filter2D(img_gt[:, :, 1], -1, psf[psf_index_h, psf_index_w, :, :, 1], borderType=cv2.BORDER_REPLICATE)
        conved_img_pad_patch_b = cv2.filter2D(img_gt[:, :, 2], -1, psf[psf_index_h, psf_index_w, :, :, 2], borderType=cv2.BORDER_REPLICATE)

        conved_img_pad_patch = np.stack((conved_img_pad_patch_r, conved_img_pad_patch_g, conved_img_pad_patch_b), axis=-1)

        img_lq = conved_img_pad_patch[self.psf_size //2: self.psf_size//2 + self.patch_size, self.psf_size//2: self.psf_size//2 + self.patch_size, :]
        img_lq = img_lq.clip(0,1)
        # add noise
        if self.opt['noise']:
            img_lq = addnoise(img_lq, self.noises[np.random.randint(0, len(self.noises))])

        # mosic demosic
        # Mosaicing
        # img_mosaiced = mosaic(img_lq, 'rggb')

        # img_mosaiced = img_mosaiced*255.0
        # img_mosaiced = np.clip(img_mosaiced, 0, 255).astype(np.uint8)

        # img_mosaiced = img_mosaiced.reshape(self.patch_size // 2, self.patch_size // 2, 2, 2).transpose(0, 2, 1, 3).reshape(self.patch_size, self.patch_size)
        # img_lq = cv2.cvtColor(img_mosaiced, cv2.COLOR_BayerRGGB2RGB_EA)/255.0


        if self.sigma:
            noise = np.random.normal(loc=0, scale=self.sigma/255.0, size=img_lq.shape)
            img_lq = img_lq + noise
            img_lq = np.clip(img_lq, 0.0, 1.0)

        img_gt = img_gt[self.psf_size //2: self.psf_size//2+ self.patch_size, self.psf_size//2: self.psf_size//2+ self.patch_size, :]


        ######################################################################################
        # h_orgin,w_orgin = self.h_orgin,self.w_orgin #3072,4096

        h_range = np.arange(psf_index_h*self.patch_size, (psf_index_h + 1) * self.patch_size, 1)
        w_range = np.arange(psf_index_w*self.patch_size, (psf_index_w + 1) * self.patch_size, 1)
        img_fld_w, img_fld_h = np.meshgrid(w_range, h_range)
        

        img_fld_h = np.expand_dims(img_fld_h, -1)
        img_fld_w = np.expand_dims(img_fld_w, -1)
        img_wz_fld = np.concatenate([img_lq, img_fld_h, img_fld_w], 2)


        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt, img_lq = img2tensor([img_gt, img_wz_fld],
                                        bgr2rgb=False,
                                        float32=True)
        

        # psf_code = np.load('psf_code.npy').reshape(12,16,52)
        # psf_patch = psf_code[psf_index_h, psf_index_w]  #52
        # psf_patch = torch.from_numpy(psf_patch).float()
        # print(psf_patch.shape)

        # normalize
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)

        return {
            'lq': img_lq,
            'gt': img_gt,
            'gt_path': gt_path,
            # 'psf_code':psf_patch
        }

    def __len__(self):
        return len(self.paths)
@DATASET_REGISTRY.register()
class BlurOnlineDatasetRGB(data.Dataset):
    """Paired image dataset for image restoration.

    Read LQ (Low Quality, e.g. LR (Low Resolution), blurry, noisy, etc) and
    GT image pairs.
    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            io_backend (dict): IO backend type and other kwarg.
            filename_tmpl (str): Template for each filename. Note that the
                template excludes the file extension. Default: '{}'.
            gt_size (int): Cropped patched size for gt patches.
            geometric_augs (bool): Use geometric augmentations.

            scale (bool): Scale, which will be added automatically.
            phase (str): 'train' or 'val'.
    """

    def __init__(self, opt):
        super(BlurOnlineDatasetRGB, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None
        self.sigma = opt.get('sigma', 3)
        self.patch_size = opt['patch_size']
        self.gt_folder = opt['dataroot_gt']
        self.noise_folder = opt['dataroot_noise']
        self.psf_folder = opt['dataroot_psf']
        self.psf_size = opt['psf_size']
        
        self.psf_paths = []
        if os.path.isdir(self.psf_folder):
            for file in os.listdir(self.psf_folder):
                self.psf_paths.append(os.path.join(self.psf_folder, file))
        else:
            self.psf_paths.append(self.psf_folder)
            
        self.h_orgin,self.w_orgin = opt['img_orgin'][:]
        self.noises = []
        for file in os.listdir(self.noise_folder):
            noise_path = os.path.join(self.noise_folder, file)
            noise = scipy.io.loadmat(noise_path)
            self.noises.append(noise)


        self.paths = paths_from_folder(self.gt_folder)


    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop('type'), **self.io_backend_opt)

        index = index % len(self.paths)
        # Load gt and lq images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.


        gt_path = self.paths[index]
        img_bytes = self.file_client.get(gt_path, 'gt')
        try:
            img_gt = imfrombytes(img_bytes, float32=True)
            img_gt = cv2.cvtColor(img_gt, cv2.COLOR_BGR2RGB)
        except:
            raise Exception("gt path {} not working".format(gt_path))


        # prepocess

        img_inv_gamma = cv2.pow(img_gt, 2.2)
        img_gt = apply_ccm(img_inv_gamma, ccm, inverse=True)
        
        # Randomly choose a color temperature
        color_temperature_index = np.random.randint(0, 4)
        wb = wb_arr[color_temperature_index, :]
        img_inv_wb = apply_wb(img_gt, wb, inverse=True)


        psf_index_h,psf_index_w = gt_path.split('/')[-1].split('_')[1:3]
        psf_index_h,psf_index_w = eval(psf_index_h),eval(psf_index_w.split('.')[0])
        psf_path = self.psf_paths[np.random.randint(0, len(self.psf_paths))]
        psf = scipy.io.loadmat(psf_path)['PSF']

        # Convolution for each color channel
        conved_img_pad_patch_r = cv2.filter2D(img_inv_wb[:, :, 0], -1, psf[psf_index_h, psf_index_w, :, :, 0], borderType=cv2.BORDER_REPLICATE)
        conved_img_pad_patch_g = cv2.filter2D(img_inv_wb[:, :, 1], -1, psf[psf_index_h, psf_index_w, :, :, 1], borderType=cv2.BORDER_REPLICATE)
        conved_img_pad_patch_b = cv2.filter2D(img_inv_wb[:, :, 2], -1, psf[psf_index_h, psf_index_w, :, :, 2], borderType=cv2.BORDER_REPLICATE)

        conved_img_pad_patch = np.stack((conved_img_pad_patch_r, conved_img_pad_patch_g, conved_img_pad_patch_b), axis=-1)

        img_lq = conved_img_pad_patch[self.psf_size //2: self.psf_size//2 + self.patch_size, self.psf_size//2: self.psf_size//2 + self.patch_size, :]
        img_lq = img_lq.clip(0,1)
        # add noise
        if self.opt['noise']:
            img_lq = addnoise(img_lq, self.noises[np.random.randint(0, len(self.noises))])

        if self.sigma:
            noise = np.random.normal(loc=0, scale=self.sigma/255.0, size=img_lq.shape)
            img_lq = img_lq + noise
            img_lq = np.clip(img_lq, 0.0, 1.0)

        img_lq = apply_wb(img_lq,wb,False)
        
        img_gt = img_gt[self.psf_size //2: self.psf_size//2+ self.patch_size, self.psf_size//2: self.psf_size//2+ self.patch_size, :]

        h_range = np.arange(psf_index_h*self.patch_size, (psf_index_h + 1) * self.patch_size, 1)
        w_range = np.arange(psf_index_w*self.patch_size, (psf_index_w + 1) * self.patch_size, 1)
        img_fld_w, img_fld_h = np.meshgrid(w_range, h_range)
        

        img_fld_h = ((img_fld_h - (self.h_orgin-1)/2) / ((self.h_orgin-1)/2)).astype(np.float32)
        img_fld_w = ((img_fld_w - (self.w_orgin-1)/2) / ((self.w_orgin-1)/2)).astype(np.float32)

        img_fld_h = np.expand_dims(img_fld_h, -1)
        img_fld_w = np.expand_dims(img_fld_w, -1)
        img_wz_fld = np.concatenate([img_lq, img_fld_h, img_fld_w], 2)

        img_gt, img_wz_fld = augment([img_gt, img_wz_fld], self.opt['use_hflip'], self.opt['use_rot'])

        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt, img_lq = img2tensor([img_gt, img_wz_fld],
                                        bgr2rgb=False,
                                        float32=True)
        
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)
            
        return {
            'lq': img_lq,
            'gt': img_gt,
            'gt_path': gt_path,
        }
    def __len__(self):
        return len(self.paths)
@DATASET_REGISTRY.register()
class BlurOnlineDatasetRGBV2(data.Dataset):
    """Paired image dataset for image restoration.

    Read LQ (Low Quality, e.g. LR (Low Resolution), blurry, noisy, etc) and
    GT image pairs.
    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            io_backend (dict): IO backend type and other kwarg.
            filename_tmpl (str): Template for each filename. Note that the
                template excludes the file extension. Default: '{}'.
            gt_size (int): Cropped patched size for gt patches.
            geometric_augs (bool): Use geometric augmentations.

            scale (bool): Scale, which will be added automatically.
            phase (str): 'train' or 'val'.
    """

    def __init__(self, opt):
        super(BlurOnlineDatasetRGBV2, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None
        self.sigma = opt.get('sigma', 3)
        self.patch_size = opt['patch_size']
        self.gt_folder = opt['dataroot_gt']
        self.noise_folder = opt['dataroot_noise']
        self.psf_folder = opt['dataroot_psf']
        self.psf_size = opt['psf_size']
        
        self.psf_paths = []
        if os.path.isdir(self.psf_folder):
            for file in os.listdir(self.psf_folder):
                self.psf_paths.append(os.path.join(self.psf_folder, file))
        else:
            self.psf_paths.append(self.psf_folder)
            
        self.h_orgin,self.w_orgin = opt['img_orgin'][:]
        self.noises = []
        for file in os.listdir(self.noise_folder):
            noise_path = os.path.join(self.noise_folder, file)
            noise = scipy.io.loadmat(noise_path)
            self.noises.append(noise)


        self.paths = paths_from_folder(self.gt_folder)


    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop('type'), **self.io_backend_opt)

        index = index % len(self.paths)
        # Load gt and lq images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.


        gt_path = self.paths[index]
        img_bytes = self.file_client.get(gt_path, 'gt')
        try:
            img_gt = imfrombytes(img_bytes, float32=True)
            img_gt = cv2.cvtColor(img_gt, cv2.COLOR_BGR2RGB)
        except:
            raise Exception("gt path {} not working".format(gt_path))

        # prepocess

        img_inv_gamma = cv2.pow(img_gt, 2.2)
        img_gt = apply_ccm(img_inv_gamma, ccm, inverse=True)
        
        # Randomly choose a color 
        wb = np.random.normal(mean_values, std_devs, size=3)
        
        img_inv_wb = apply_wb(img_gt, wb, inverse=True)


        psf_index_h,psf_index_w = gt_path.split('/')[-1].split('_')[1:3]
        psf_index_h,psf_index_w = eval(psf_index_h),eval(psf_index_w.split('.')[0])
        psf_path = self.psf_paths[np.random.randint(0, len(self.psf_paths))]
        psf = scipy.io.loadmat(psf_path)['PSF']

        # Convolution for each color channel
        conved_img_pad_patch_r = cv2.filter2D(img_inv_wb[:, :, 0], -1, psf[psf_index_h, psf_index_w, :, :, 0], borderType=cv2.BORDER_REPLICATE)
        conved_img_pad_patch_g = cv2.filter2D(img_inv_wb[:, :, 1], -1, psf[psf_index_h, psf_index_w, :, :, 1], borderType=cv2.BORDER_REPLICATE)
        conved_img_pad_patch_b = cv2.filter2D(img_inv_wb[:, :, 2], -1, psf[psf_index_h, psf_index_w, :, :, 2], borderType=cv2.BORDER_REPLICATE)

        conved_img_pad_patch = np.stack((conved_img_pad_patch_r, conved_img_pad_patch_g, conved_img_pad_patch_b), axis=-1)

        img_lq = conved_img_pad_patch[self.psf_size //2: self.psf_size//2 + self.patch_size, self.psf_size//2: self.psf_size//2 + self.patch_size, :]
        img_lq = img_lq.clip(0,1)
        # add noise
        if self.opt['noise']:
            img_lq = addnoise(img_lq, self.noises[np.random.randint(0, len(self.noises))])

        if self.sigma:
            noise = np.random.normal(loc=0, scale=self.sigma/255.0, size=img_lq.shape)
            img_lq = img_lq + noise
            img_lq = np.clip(img_lq, 0.0, 1.0)

        
        img_gt = img_inv_wb[self.psf_size //2: self.psf_size//2+ self.patch_size, self.psf_size//2: self.psf_size//2+ self.patch_size, :]

        h_range = np.arange(psf_index_h*self.patch_size, (psf_index_h + 1) * self.patch_size, 1)
        w_range = np.arange(psf_index_w*self.patch_size, (psf_index_w + 1) * self.patch_size, 1)
        img_fld_w, img_fld_h = np.meshgrid(w_range, h_range)
        

        img_fld_h = ((img_fld_h - (self.h_orgin-1)/2) / ((self.h_orgin-1)/2)).astype(np.float32)
        img_fld_w = ((img_fld_w - (self.w_orgin-1)/2) / ((self.w_orgin-1)/2)).astype(np.float32)

        img_fld_h = np.expand_dims(img_fld_h, -1)
        img_fld_w = np.expand_dims(img_fld_w, -1)
        img_wz_fld = np.concatenate([img_lq, img_fld_h, img_fld_w], 2)

        img_gt, img_wz_fld = augment([img_gt, img_wz_fld], self.opt['use_hflip'], self.opt['use_rot'])

        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt, img_lq = img2tensor([img_gt, img_wz_fld],
                                        bgr2rgb=False,
                                        float32=True)
        
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)
            
        return {
            'lq': img_lq,
            'gt': img_gt,
            'gt_path': gt_path,
        }
    def __len__(self):
        return len(self.paths)
