import torch
from collections import OrderedDict
from os import path as osp
import os
from fewlens.utils import img2tensor, tensor2img, imwrite
from tqdm import tqdm
from fewlens.data.transforms import paired_random_crop
from fewlens.archs import build_network
from fewlens.losses import build_loss
from .base_model import BaseModel
from fewlens.metrics import calculate_metric
from fewlens.utils import get_root_logger, imwrite, tensor2img
from fewlens.utils.registry import MODEL_REGISTRY
import torchvision.utils as tvu
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.checkpoint import checkpoint
import cv2
from fewlens.utils.isp import apply_ccm
ccm = np.array([[ 1.93994141, -0.73925781, -0.20068359],
                [-0.28857422,  1.59741211, -0.30883789],
                [-0.0078125 , -0.45654297,  1.46435547]])

@MODEL_REGISTRY.register()
class RestorationModel(BaseModel):
    def __init__(self, opt):
        super().__init__(opt)
        self.queue_size = opt.get('queue_size', 180)

        # define network g
        self.net_g = build_network(self.opt['network_g'])

        self.net_g = self.model_to_device(self.net_g)
        self.print_network(self.net_g)
        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_g', None)
        logger = get_root_logger()
        if load_path is not None:
            logger.info(f'Loading net_g from {load_path}')
            self.load_network(self.net_g, load_path,
                              self.opt['path']['strict_load'])

        frozen_module_keywords = self.opt['network_g'].get('frozen_module_keywords', None)
        if frozen_module_keywords is not None:
            for name, module in self.net_g.named_modules():
                for fkw in frozen_module_keywords:
                    if fkw in name:
                        for p in module.parameters():
                            p.requires_grad = False
                        break

        if self.is_train:
            self.init_training_settings()

    def feed_train_data(self, data):
        self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)
        # if 'psf_bases' in data:
        #     self.psf_bases = data['psf_bases'].to(self.device)

        if self.mixing_flag:
            self.gt, self.lq = self.mixing_augmentation(self.gt, self.lq)


    def feed_data(self, data):

        self.OptParam = {}
        if 'lq' in data:
            self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)
        # if 'psf_bases' in data:
        #     self.psf_bases = data['psf_bases'].to(self.device)
            

    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params = []
        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
            else:
                logger = get_root_logger()
                logger.warning(f'Params {k} will not be optimized.')

        optim_type = train_opt['optim_g'].pop('type')
        self.optimizer_g = self.get_optimizer(optim_type, optim_params,
                                              **train_opt['optim_g'])
        self.optimizers.append(self.optimizer_g)

    # @torch.no_grad()
    # def _dequeue_and_enqueue(self):
    #     """It is the training pair pool for increasing the diversity in a batch.

    #     Batch processing limits the diversity of synthetic degradations in a batch. For example, samples in a
    #     batch could not have different resize scaling factors. Therefore, we employ this training pair pool
    #     to increase the degradation diversity in a batch.
    #     """
    #     # initialize
    #     b, c, h, w = self.lq.size()
    #     if not hasattr(self, 'queue_lr'):
    #         assert self.queue_size % b == 0, f'queue size {self.queue_size} should be divisible by batch size {b}'
    #         self.queue_lr = torch.zeros(self.queue_size, c, h, w).cuda()
    #         _, c, h, w = self.gt.size()
    #         self.queue_gt = torch.zeros(self.queue_size, c, h, w).cuda()
    #         self.queue_ptr = 0
    #     if self.queue_ptr == self.queue_size:  # the pool is full
    #         # do dequeue and enqueue
    #         # shuffle
    #         idx = torch.randperm(self.queue_size)
    #         self.queue_lr = self.queue_lr[idx]
    #         self.queue_gt = self.queue_gt[idx]
    #         # get first b samples
    #         lq_dequeue = self.queue_lr[0:b, :, :, :].clone()
    #         gt_dequeue = self.queue_gt[0:b, :, :, :].clone()
    #         # update the queue
    #         self.queue_lr[0:b, :, :, :] = self.lq.clone()
    #         self.queue_gt[0:b, :, :, :] = self.gt.clone()

    #         self.lq = lq_dequeue
    #         self.gt = gt_dequeue
    #     else:
    #         # only do enqueue
    #         self.queue_lr[self.queue_ptr:self.queue_ptr +
    #                       b, :, :, :] = self.lq.clone()
    #         self.queue_gt[self.queue_ptr:self.queue_ptr +
    #                       b, :, :, :] = self.gt.clone()
    #         self.queue_ptr = self.queue_ptr + b

    def init_training_settings(self):
        self.net_g.train()
        train_opt = self.opt['train']

        self.ema_decay = train_opt.get('ema_decay', 0)
        # define losses
        # define losses
        if train_opt.get('pixel_opt'):
            self.cri_pix = build_loss(train_opt['pixel_opt']).to(self.device)
        else:
            self.cri_pix = None
        # define losses
        if train_opt.get('pixel_SRN_opt'):
            self.cri_pix_SRN = build_loss(train_opt['pixel_SRN_opt']).to(self.device)
        else:
            self.cri_pix_SRN = None
            
        if train_opt.get('latent_opt'):
            self.cri_latent = build_loss(train_opt['latent_opt']).to(self.device)
        else:
            self.cri_latent = None
        if train_opt.get('perceptual_opt'):
            self.cri_perceptual = build_loss(train_opt['perceptual_opt']).to(
                self.device)
        else:
            self.cri_perceptual = None
            
        if train_opt.get("wavelet_opt"):
            self.cri_wavelet = train_opt['wavelet_opt'].get('use_loss', False)
        else:
            self.cri_wavelet = False
        if train_opt.get('fft_opt'):
            self.cri_fft =  build_loss(train_opt['fft_opt']).to(self.device)
        else:
            self.cri_fft = None
        if train_opt.get('pixel_L2_wz_Perceptual_opt'):
            self.cri_L2_wz_Perceptual = build_loss(train_opt['pixel_L2_wz_Perceptual_opt']).to(self.device)
        else:
            self.cri_L2_wz_Perceptual = None
            

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def get_current_visuals(self):
        vis_samples = 4
        out_dict = OrderedDict()
        out_dict['lq'] = self.lq.detach().cpu()[:vis_samples,:3]

        if hasattr(self, 'output'):
            out_dict['output'] = self.output.detach().cpu()[:vis_samples]
        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt.detach().cpu()[:vis_samples]
        if hasattr(self, 'hq_rec'):
            out_dict['hq_rec'] = self.hq_rec.detach().cpu()[:vis_samples]
        if hasattr(self, 'lq_rec'):
            out_dict['lq_rec'] = self.lq_rec.detach().cpu()[:vis_samples]

        return out_dict

    def optimize_parameters(self, current_iter):

        train_opt = self.opt['train']

        self.optimizer_g.zero_grad()
        l_total = 0
        loss_dict = OrderedDict()
        # cv2.imwrite('coc_lq.png', tensor2img(self.coc_lq))
        # if self.psf_bases is not None:
        #     preds = self.net_g(self.lq, self.psf_bases)
        # else:       
        preds = self.net_g(self.lq)
        if isinstance(preds, tuple):
            preds = list(preds)
        if not isinstance(preds, list):
            preds = [preds]

        self.output = preds[0] 
        
        frozen_module_keywords = None
        # self.opt['network_g'].get('frozen_module_keywords', None)
        if self.cri_L2_wz_Perceptual:
            L2_wz_Perceptual = self.cri_L2_wz_Perceptual(self.output,self.gt)
            l_total += L2_wz_Perceptual
            loss_dict['L2_wz_Perceptual'] = L2_wz_Perceptual
            
            
        if self.cri_pix_SRN:
            # l_pix_coc = self.cri_pix(self.coc_lq, self.coc_gt*self.lq_Af/self.gt_Af)
            # SRN
            l_pix_gt = self.cri_pix_SRN(preds, self.gt)
            # pixel
            # l_pix_gt = self.cri_pix(self.output, self.gt)
            l_total += l_pix_gt
            loss_dict['l_pix_SRN'] = l_pix_gt
        if self.cri_fft:
            l_fft = self.cri_fft(preds, self.gt)
            l_total+=l_fft
            loss_dict['l_fft'] = l_fft
        if self.cri_pix:

            l_pix_gt = self.cri_pix(self.output, self.gt)
            # pixel
            # l_pix_gt = self.cri_pix(self.output, self.gt)
            l_total += l_pix_gt
            loss_dict['l_pix'] = l_pix_gt
            
        if self.cri_latent:
            # SRN
            latent_blur = self.net_g.module.get_latent(self.lq)
            latent_gt = self.net_g.module.get_latent(self.gt)
            l_latent_gt = self.cri_latent(latent_blur, latent_gt)
          
            l_total += l_latent_gt
            loss_dict['l_latent'] = l_latent_gt
            
        # perceptual loss
        if self.cri_perceptual:
            l_percep_gt, l_style_gt = self.cri_perceptual(self.output, self.gt)
            if l_percep_gt is not None:
                l_total += l_percep_gt
                loss_dict['l_percep'] = l_percep_gt
            if l_style_gt is not None:
                l_total += l_style_gt
                loss_dict['l_style'] = l_style_gt
                
        # wavelet loss
        if self.cri_wavelet:
            l_wavelet = self.net_g.module.get_wavelet_loss()
            l_total += l_wavelet
            loss_dict['l_wavelet'] = l_wavelet

        l_total.backward()
        self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

    def test(self):
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                self.output = self.net_g_ema(self.lq)
        else:
            self.net_g.eval()
            with torch.no_grad():
                # b, c, h, w = self.lq.shape
                # self.lq = self.lq[:,:,h%64,w%64]
                b, c, h, w = self.lq.shape
                if h*w<1024*2024:
                    self.output = self.net_g(self.lq)[0]

                else:
                    patch_size = 1024
                    stride = 1024
                    if self.opt['network_g'].get('type') == 'OneFAll':
                        patch_size = 256
                        stride = 256


                    output = torch.zeros((b,3,h,w)).to(self.device)
                    weight_map = torch.zeros((b,3,h,w)).to(self.device)

                    for y in range(0, h, stride):
                        for x in range(0, w, stride):
                            y_end = min(y + patch_size, h)
                            x_end = min(x + patch_size, w)
                            y_start = max(y_end - patch_size, 0)
                            x_start = max(x_end - patch_size, 0)

                            patch = self.lq[:, :, y_start:y_end, x_start:x_end]

                            patch_output = self.net_g(patch)[0]

                            output[:, :, y_start:y_end, x_start:x_end] += patch_output
                            weight_map[:, :, y_start:y_end, x_start:x_end] += 1

                    self.output = output / weight_map

            self.net_g.train()

    def test_selfensemble(self):
        # TODO: to be tested
        # 8 augmentations
        # modified from https://github.com/thstkdgus35/EDSR-PyTorch

        def _transform(v, op):
            # if self.precision != 'single': v = v.float()
            v2np = v.data.cpu().numpy()
            if op == 'v':
                tfnp = v2np[:, :, :, ::-1].copy()
            elif op == 'h':
                tfnp = v2np[:, :, ::-1, :].copy()
            elif op == 't':
                tfnp = v2np.transpose((0, 1, 3, 2)).copy()

            ret = torch.Tensor(tfnp).to(self.device)
            # if self.precision == 'half': ret = ret.half()

            return ret

        # prepare augmented data
        lq_list = [self.lq]
        for tf in 'v', 'h', 't':
            lq_list.extend([_transform(t, tf) for t in lq_list])

        # inference
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                out_list = [self.net_g_ema(aug) for aug in lq_list]
        else:
            self.net_g.eval()
            with torch.no_grad():
                out_list = [self.net_g_ema(aug) for aug in lq_list]
            self.net_g.train()

        # merge results
        for i in range(len(out_list)):
            if i > 3:
                out_list[i] = _transform(out_list[i], 't')
            if i % 4 > 1:
                out_list[i] = _transform(out_list[i], 'h')
            if (i % 4) % 2 == 1:
                out_list[i] = _transform(out_list[i], 'v')
        output = torch.cat(out_list, dim=0)

        self.output = output.mean(dim=0, keepdim=True)

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img):
        if self.opt['rank'] == 0:
            self.nondist_validation(dataloader, current_iter, tb_logger, save_img)

    def nondist_validation(self, dataloader, current_iter, tb_logger,
                           save_img):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        if with_metrics:
            if not hasattr(self, 'metric_results'):  # only execute in the first run
                self.metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
            # initialize the best metric results for each dataset_name (supporting multiple validation datasets)
            self._initialize_best_metric_results(dataset_name)
        # zero self.metric_results
        if with_metrics:
            self.metric_results = {metric: 0 for metric in self.metric_results}

        metric_data = dict()
        pbar = tqdm(total=len(dataloader), unit='image')

        for idx, val_data in enumerate(dataloader):
            img_name = osp.splitext(osp.basename(val_data['gt_path'][0]))[0]
            self.feed_data(val_data)
            self.test()

            visuals = self.get_current_visuals()
            # todo
            output = tensor2img([visuals['output']])
            metric_data['img'] = output

            if 'gt' in visuals:
                gt_img = tensor2img([visuals['gt']])
                metric_data['img2'] = gt_img
                del self.gt

            # tentative for out of GPU memory

            del self.output

            torch.cuda.empty_cache()

            if save_img:
                if self.opt['is_train']:
                    save_img_path = osp.join(self.opt['path']['visualization'], 'image_results',
                                             f'{current_iter}',
                                             f'{img_name}.png')
                else:
                    if self.opt['val']['suffix']:
                        save_img_path = osp.join(
                            self.opt['path']['visualization'], dataset_name,
                            f'{img_name}_{self.opt["val"]["suffix"]}.png')
                    else:
                        save_img_path = osp.join(self.opt['path']['visualization'], 'image_results',
                                             f'{current_iter}',
                                             f'{img_name}.png')
                # # img2 isp 
                if visuals['output'].shape[0] == 3:
                    output_isp = visuals['output'].permute(1, 2, 0).cpu().numpy()
                else:
                    output_isp = visuals['output'][0].permute(1, 2, 0).cpu().numpy()
                    
                # print(output_isp)
                if self.opt['val']['isp']:
                    output_isp = apply_ccm(output_isp, ccm, inverse=False)
                    output_isp = cv2.pow(output_isp, 1/2.2)
                output_isp = np.clip(output_isp,0,1)*255.0
                output_isp = output_isp.astype(np.uint8)
                output = cv2.cvtColor(output_isp, cv2.COLOR_RGB2BGR)
                imwrite(output, save_img_path)


            if with_metrics:
                # calculate metrics
                for name, opt_ in self.opt['val']['metrics'].items():
                    self.metric_results[name] += calculate_metric(metric_data, opt_)

            pbar.update(1)
            pbar.set_description(f'Test {img_name}')

        pbar.close()

        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= (idx + 1)
                # update the best metric result
                self._update_best_metric_result(dataset_name, metric, self.metric_results[metric], current_iter)

            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)

    def _log_validation_metric_values(self, current_iter, dataset_name, tb_logger):
        log_str = f'Validation {dataset_name}\n'
        for metric, value in self.metric_results.items():
            log_str += f'\t # {metric}: {value:.4f}'
            if hasattr(self, 'best_metric_results'):
                log_str += (f'\tBest: {self.best_metric_results[dataset_name][metric]["val"]:.4f} @ '
                            f'{self.best_metric_results[dataset_name][metric]["iter"]} iter')
            log_str += '\n'

        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f'metrics/{dataset_name}/{metric}', value, current_iter)


    def save(self, epoch, current_iter):
        self.save_network(self.net_g, 'net_g', current_iter)
        self.save_training_state(epoch, current_iter)
