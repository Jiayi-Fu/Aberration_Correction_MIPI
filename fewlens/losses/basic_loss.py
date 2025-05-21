import torch
from torch import nn as nn
from torch.nn import functional as F

from fewlens.archs.vgg_arch import VGGFeatureExtractor
from fewlens.utils.registry import LOSS_REGISTRY
from .loss_util import weighted_loss
import pyiqa
import numpy as np
from torchvision import models
from functools import partial

_reduction_modes = ['none', 'mean', 'sum']


@weighted_loss
def l1_loss(pred, target):
    return F.l1_loss(pred, target, reduction='none')


@weighted_loss
def mse_loss(pred, target):
    return F.mse_loss(pred, target, reduction='none')


@weighted_loss
def charbonnier_loss(pred, target, eps=1e-12):
    return torch.sqrt((pred - target)**2 + eps)


@LOSS_REGISTRY.register()
class L1Loss(nn.Module):
    """L1 (mean absolute error, MAE) loss.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(L1Loss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise weights. Default: None.
        """
        return self.loss_weight * l1_loss(pred, target, weight, reduction=self.reduction)

@LOSS_REGISTRY.register()
class CharFreqLoss(nn.Module):
    """L1 (mean absolute error, MAE) loss of fft.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(CharFreqLoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction
        self.l1_loss = CharbonnierLoss(loss_weight, reduction)

    def forward(self, pred, target):
        diff = torch.fft.rfft2(pred) - torch.fft.rfft2(target)
        loss = torch.mean(torch.abs(diff))
        # print(loss)
        return self.loss_weight * loss * 0.01 + self.l1_loss(pred, target)
    
@LOSS_REGISTRY.register()
class MSELoss(nn.Module):
    """MSE (L2) loss.

    Args:
        loss_weight (float): Loss weight for MSE loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(MSELoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise weights. Default: None.
        """
        return self.loss_weight * mse_loss(pred, target, weight, reduction=self.reduction)


@LOSS_REGISTRY.register()
class CharbonnierLoss(nn.Module):
    """Charbonnier loss (one variant of Robust L1Loss, a differentiable
    variant of L1Loss).

    Described in "Deep Laplacian Pyramid Networks for Fast and Accurate
        Super-Resolution".

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
        eps (float): A value used to control the curvature near zero. Default: 1e-12.
    """

    def __init__(self, loss_weight=1.0, reduction='mean', eps=1e-12):
        super(CharbonnierLoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction
        self.eps = eps

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise weights. Default: None.
        """
        return self.loss_weight * charbonnier_loss(pred, target, weight, eps=self.eps, reduction=self.reduction)


@LOSS_REGISTRY.register()
class WeightedTVLoss(L1Loss):
    """Weighted TV loss.

    Args:
        loss_weight (float): Loss weight. Default: 1.0.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        if reduction not in ['mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: mean | sum')
        super(WeightedTVLoss, self).__init__(loss_weight=loss_weight, reduction=reduction)

    def forward(self, pred, weight=None):
        if weight is None:
            y_weight = None
            x_weight = None
        else:
            y_weight = weight[:, :, :-1, :]
            x_weight = weight[:, :, :, :-1]

        y_diff = super().forward(pred[:, :, :-1, :], pred[:, :, 1:, :], weight=y_weight)
        x_diff = super().forward(pred[:, :, :, :-1], pred[:, :, :, 1:], weight=x_weight)

        loss = x_diff + y_diff

        return loss

@LOSS_REGISTRY.register()
class LPIPSLoss(nn.Module):
    """LPIPS loss with vgg backbone.
    """
    def __init__(self, loss_weight = 1.0):
        super(LPIPSLoss, self).__init__()
        self.model = pyiqa.create_metric('lpips-vgg', as_loss=True)
        self.loss_weight = loss_weight

    def forward(self, x, gt):
        return self.model(x, gt) * self.loss_weight, None
@LOSS_REGISTRY.register()
class PerceptualLoss(nn.Module):
    """Perceptual loss with commonly used style loss.

    Args:
        layer_weights (dict): The weight for each layer of vgg feature.
            Here is an example: {'conv5_4': 1.}, which means the conv5_4
            feature layer (before relu5_4) will be extracted with weight
            1.0 in calculating losses.
        vgg_type (str): The type of vgg network used as feature extractor.
            Default: 'vgg19'.
        use_input_norm (bool):  If True, normalize the input image in vgg.
            Default: True.
        range_norm (bool): If True, norm images with range [-1, 1] to [0, 1].
            Default: False.
        perceptual_weight (float): If `perceptual_weight > 0`, the perceptual
            loss will be calculated and the loss will multiplied by the
            weight. Default: 1.0.
        style_weight (float): If `style_weight > 0`, the style loss will be
            calculated and the loss will multiplied by the weight.
            Default: 0.
        criterion (str): Criterion used for perceptual loss. Default: 'l1'.
    """

    def __init__(self,
                 layer_weights,
                 vgg_type='vgg19',
                 use_input_norm=True,
                 range_norm=False,
                 perceptual_weight=1.0,
                 style_weight=0.,
                 criterion='l1'):
        super(PerceptualLoss, self).__init__()
        self.perceptual_weight = perceptual_weight
        self.style_weight = style_weight
        self.layer_weights = layer_weights
        self.vgg = VGGFeatureExtractor(
            layer_name_list=list(layer_weights.keys()),
            vgg_type=vgg_type,
            use_input_norm=use_input_norm,
            range_norm=range_norm)

        self.criterion_type = criterion
        if self.criterion_type == 'l1':
            self.criterion = torch.nn.L1Loss()
        elif self.criterion_type == 'l2':
            self.criterion = torch.nn.MSELoss()
        elif self.criterion_type == 'fro':
            self.criterion = None
        else:
            raise NotImplementedError(f'{criterion} criterion has not been supported.')

    def forward(self, x, gt):
        """Forward function.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).
            gt (Tensor): Ground-truth tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """
        # extract vgg features
        x_features = self.vgg(x)
        gt_features = self.vgg(gt.detach())

        # calculate perceptual loss
        if self.perceptual_weight > 0:
            percep_loss = 0
            for k in x_features.keys():
                if self.criterion_type == 'fro':
                    percep_loss += torch.norm(x_features[k] - gt_features[k], p='fro') * self.layer_weights[k]
                else:
                    percep_loss += self.criterion(x_features[k], gt_features[k]) * self.layer_weights[k]
            percep_loss *= self.perceptual_weight
        else:
            percep_loss = None

        # calculate style loss
        if self.style_weight > 0:
            style_loss = 0
            for k in x_features.keys():
                if self.criterion_type == 'fro':
                    style_loss += torch.norm(
                        self._gram_mat(x_features[k]) - self._gram_mat(gt_features[k]), p='fro') * self.layer_weights[k]
                else:
                    style_loss += self.criterion(self._gram_mat(x_features[k]), self._gram_mat(
                        gt_features[k])) * self.layer_weights[k]
            style_loss *= self.style_weight
        else:
            style_loss = None

        return percep_loss, style_loss

    def _gram_mat(self, x):
        """Calculate Gram matrix.

        Args:
            x (torch.Tensor): Tensor with shape of (n, c, h, w).

        Returns:
            torch.Tensor: Gram matrix.
        """
        n, c, h, w = x.size()
        features = x.view(n, c, w * h)
        features_t = features.transpose(1, 2)
        gram = features.bmm(features_t) / (c * h * w)
        return gram
@LOSS_REGISTRY.register()
class SRNL2_loss(nn.Module):
    def __init__(self, loss_weight=1.0, reduction='mean', toY=False):
        super(SRNL2_loss, self).__init__()
        # self.eps = e
        assert reduction == 'mean'
        self.loss_weight = loss_weight
        # self.scale = 10 / np.log(10)
        self.scale = 1
        self.toY = toY
        self.coef = torch.tensor([65.481, 128.553, 24.966]).reshape(1, 3, 1, 1)
        self.first = True
        self.resize = partial(F.interpolate, mode='area', recompute_scale_factor=True)
        # self.scale = 1e2 / np.log(1e2)

    def forward(self, batch_p, batch_l):
        # print(batch_p[0].shape,batch_p[1].shape,batch_p[2].shape)
        assert batch_p[0].shape[0] == batch_l.shape[0]
        device = batch_p[0].device
        b, c, h, w = batch_p[0].shape
        # self.loss_weight * self.scale * torch.log(((pred - target) ** 2).mean(dim=(1, 2, 3)) + 1e-8).mean()
        # loss = l1_loss(batch_p[0],batch_l)
        loss = 0.0
        ####################################################### multi-scale
        for scale in range(len(batch_p)):
            scale_num = 1/(2**(scale))
            loss += scale_num * mse_loss(batch_p[scale],self.resize(input=batch_l, scale_factor=scale_num))
        # loss += 0.25 * l1_loss(batch_p[2],self.resize(input=batch_l, scale_factor=0.25))
        # loss += 0.125 * l1_loss(batch_p[3],self.resize(input=batch_l, scale_factor=0.125))
        
        # loss += 0.25 * self.loss_weight * self.scale * (
        #     ((batch_p[2] - self.resize(input=batch_l, scale_factor=0.25)) ** 2).mean(dim=(1, 2, 3)) + 1e-8).mean()
        # loss += 0.125 * self.loss_weight * self.scale*(
        #     ((batch_p[3] - self.resize(input=batch_l, scale_factor=0.125)) ** 2).mean(dim=(1, 2, 3)) + 1e-8).mean()
        #################################################################
        # TODO scale100
        # js_loss = torch.tensor([0.7, 0.5, 0.3]).to(device).dot(
        #     torch.stack([js_div(batch_p[1], F.interpolate(batch_p[0], size=(h // 2, w // 2), mode="nearest")),
        #                  js_div(batch_p[2], F.interpolate(batch_p[0], size=(h // 4, w // 4), mode="nearest")),
        #                  js_div(batch_p[3], F.interpolate(batch_p[0], size=(h // 8, w // 8), mode="nearest"))]))

        # loss += js_loss
        return loss*self.loss_weight
@LOSS_REGISTRY.register()
class SRN_loss(nn.Module):
    def __init__(self, loss_weight=1.0, reduction='mean', toY=False):
        super(SRN_loss, self).__init__()
        # self.eps = e
        assert reduction == 'mean'
        self.loss_weight = loss_weight
        # self.scale = 10 / np.log(10)
        self.scale = 1
        self.toY = toY
        self.coef = torch.tensor([65.481, 128.553, 24.966]).reshape(1, 3, 1, 1)
        self.first = True
        self.resize = partial(F.interpolate, mode='area', recompute_scale_factor=True)
        # self.scale = 1e2 / np.log(1e2)

    def forward(self, batch_p, batch_l):
        # print(batch_p[0].shape,batch_p[1].shape,batch_p[2].shape)
        assert batch_p[0].shape[0] == batch_l.shape[0]
        device = batch_p[0].device
        b, c, h, w = batch_p[0].shape
        # self.loss_weight * self.scale * torch.log(((pred - target) ** 2).mean(dim=(1, 2, 3)) + 1e-8).mean()
        # loss = l1_loss(batch_p[0],batch_l)
        loss = 0.0
        ####################################################### multi-scale
        for scale in range(len(batch_p)):
            scale_num = 1/(2**(scale))
            loss += scale_num * l1_loss(batch_p[scale],self.resize(input=batch_l, scale_factor=scale_num))
        # loss += 0.25 * l1_loss(batch_p[2],self.resize(input=batch_l, scale_factor=0.25))
        # loss += 0.125 * l1_loss(batch_p[3],self.resize(input=batch_l, scale_factor=0.125))
        
        # loss += 0.25 * self.loss_weight * self.scale * (
        #     ((batch_p[2] - self.resize(input=batch_l, scale_factor=0.25)) ** 2).mean(dim=(1, 2, 3)) + 1e-8).mean()
        # loss += 0.125 * self.loss_weight * self.scale*(
        #     ((batch_p[3] - self.resize(input=batch_l, scale_factor=0.125)) ** 2).mean(dim=(1, 2, 3)) + 1e-8).mean()
        #################################################################
        # TODO scale100
        # js_loss = torch.tensor([0.7, 0.5, 0.3]).to(device).dot(
        #     torch.stack([js_div(batch_p[1], F.interpolate(batch_p[0], size=(h // 2, w // 2), mode="nearest")),
        #                  js_div(batch_p[2], F.interpolate(batch_p[0], size=(h // 4, w // 4), mode="nearest")),
        #                  js_div(batch_p[3], F.interpolate(batch_p[0], size=(h // 8, w // 8), mode="nearest"))]))

        # loss += js_loss
        return loss*self.loss_weight
    
def rfft(x, d):
    t = torch.fft.rfft2(x, dim = (-d))
    return torch.stack((t.real, t.imag), -1)
@LOSS_REGISTRY.register()
class Multi_FFT_loss(nn.Module):
    def __init__(self, loss_weight=1.0, reduction='mean', toY=False):
        super(Multi_FFT_loss, self).__init__()
        # self.eps = e
        assert reduction == 'mean'
        self.loss_weight = loss_weight
        # self.scale = 10 / np.log(10)
        self.scale = 1
        self.first = True
        self.resize = partial(F.interpolate, mode='area', recompute_scale_factor=True)
        # self.scale = 1e2 / np.log(1e2)

    def forward(self, batch_p, batch_l):
        # print(batch_p[0].shape,batch_p[1].shape,batch_p[2].shape)
        assert batch_p[0].shape[0] == batch_l.shape[0]
        device = batch_p[0].device
        b, c, h, w = batch_p[0].shape
        # self.loss_weight * self.scale * torch.log(((pred - target) ** 2).mean(dim=(1, 2, 3)) + 1e-8).mean()
        # loss = l1_loss(batch_p[0],batch_l)
        loss = 0.0
        ####################################################### multi-scale
        for scale in range(len(batch_p)):
            scale_num = 1/(2**(scale))
            label = self.resize(input=batch_l, scale_factor=scale_num)
            predict = batch_p[scale]
            
            # label_fft = torch.fft.rfft(label, signal_ndim=2, normalized=False, onesided=False)
            label_fft = rfft(label, 2)
            pred_fft = rfft(predict,2)
            # pred_fft = torch.rfft(predct, signal_ndim=2, normalized=False, onesided=False)
            
            loss += scale_num * l1_loss(label_fft,pred_fft)
   
        # loss += js_loss
        return loss*self.loss_weight
    
# using ImageNet values
def normalize_tensor_transform(output, label):
    output_norm = torch.zeros_like(output)
    output_norm[:, 0, ...] = (output[:, 0, ...] - 0.485) / 0.229
    output_norm[:, 1, ...] = (output[:, 1, ...] - 0.456) / 0.224
    output_norm[:, 2, ...] = (output[:, 2, ...] - 0.406) / 0.225

    label_norm = torch.zeros_like(label)
    label_norm[:, 0, ...] = (label[:, 0, ...] - 0.485) / 0.229
    label_norm[:, 1, ...] = (label[:, 1, ...] - 0.456) / 0.224
    label_norm[:, 2, ...] = (label[:, 2, ...] - 0.406) / 0.225

    return output_norm, label_norm

# class PerceptualLoss(nn.Module):
#     def __init__(self):
#         super(PerceptualLoss, self).__init__()
#         features = models.vgg16(pretrained=True).features
#         self.to_relu_1_2 = nn.Sequential()
#         self.to_relu_2_2 = nn.Sequential()
#         self.to_relu_3_3 = nn.Sequential()
#         self.to_relu_4_3 = nn.Sequential()
        
#         for x in range(4):
#             self.to_relu_1_2.add_module(str(x), features[x])
#         for x in range(4, 9):
#             self.to_relu_2_2.add_module(str(x), features[x])
#         for x in range(9, 16):
#             self.to_relu_3_3.add_module(str(x), features[x])
#         for x in range(16, 23):
#             self.to_relu_4_3.add_module(str(x), features[x])    

#         self.mse_loss = nn.MSELoss()
#         # don't need the gradients, just want the features
#         for param in self.parameters():
#             param.requires_grad = False

#     def _gram(self, x):
#         (bs, ch, h, w) = x.size()
#         f = x.view(bs, ch, w*h)
#         f_T = f.transpose(1, 2)
#         G = f.bmm(f_T) / (ch * h * w)
#         return G

#     def forward(self, pred_img, targ_img):
#         h_relu_1_2_pred_img = self.to_relu_1_2(pred_img)
#         h_relu_1_2_targ_img = self.to_relu_1_2(targ_img)
#         style_loss_1_2 = self.mse_loss(self._gram(h_relu_1_2_pred_img), self._gram(h_relu_1_2_targ_img))
        
#         h_relu_2_2_pred_img = self.to_relu_2_2(h_relu_1_2_pred_img)
#         h_relu_2_2_targ_img = self.to_relu_2_2(h_relu_1_2_targ_img)
#         style_loss_2_2 = self.mse_loss(self._gram(h_relu_2_2_pred_img), self._gram(h_relu_2_2_targ_img))
        
#         h_relu_3_3_pred_img = self.to_relu_3_3(h_relu_2_2_pred_img)
#         h_relu_3_3_targ_img = self.to_relu_3_3(h_relu_2_2_targ_img)
#         style_loss_3_3 = self.mse_loss(self._gram(h_relu_3_3_pred_img), self._gram(h_relu_3_3_targ_img))

#         h_relu_4_3_pred_img = self.to_relu_4_3(h_relu_3_3_pred_img)
#         h_relu_4_3_targ_img = self.to_relu_4_3(h_relu_3_3_targ_img)
#         style_loss_4_3 = self.mse_loss(self._gram(h_relu_4_3_pred_img), self._gram(h_relu_4_3_targ_img))
        
#         style_loss_tol = style_loss_1_2 + style_loss_2_2 + style_loss_3_3 + style_loss_4_3
#         # content loss (h_relu_2_2)
#         content_loss_tol = style_loss_2_2
#         return style_loss_tol, content_loss_tol

@LOSS_REGISTRY.register()
class L2_wz_Perceptual(nn.Module):
    def __init__(self, style_weight,content_weight, reduction='mean',):
        super(L2_wz_Perceptual, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.per_loss = PerceptualLoss()
        self.STYLE_WEIGHT = style_weight
        self.CONTENT_WEIGHT = content_weight

    def forward(self, out_images, target_images):
        # MSELoss
        image_loss = self.mse_loss(out_images, target_images)
        # Perceptual Loss
        out_images_norm, target_images_norm = normalize_tensor_transform(out_images, target_images)
        style_loss, content_loss = self.per_loss(out_images_norm, target_images_norm)
        # print(style_loss.data, content_loss.data)
        return image_loss + self.STYLE_WEIGHT * style_loss.data + self.CONTENT_WEIGHT * content_loss.data
