


# import torch.nn as nn
# import torch
# import torch.nn.functional as F
# from torch import nn
# from torch.nn.modules.utils import _pair, _quadruple
# # from fewlens.utils.registry import ARCH_REGISTRY
# import numpy as np

# import torch.nn as nn
# import torch
# import torch.nn.functional as F
# from torch import nn
# from torch.nn.modules.utils import _pair, _quadruple
# import numpy as np
# from fewlens.utils.registry import ARCH_REGISTRY
# class MedianPool2d(nn.Module):
#     """ Median pool (usable as median filter when stride=1) module.

#     Args:
#          kernel_size: size of pooling kernel, int or 2-tuple
#          stride: pool stride, int or 2-tuple
#          padding: pool padding, int or 4-tuple (l, r, t, b) as in pytorch F.pad
#          same: override padding and enforce same padding, boolean
#     """
#     def __init__(self, kernel_size=3, stride=1, padding=0, same=False):
#         super(MedianPool2d, self).__init__()
#         self.k = _pair(kernel_size)
#         self.stride = _pair(stride)
#         self.padding = _quadruple(padding)  # convert to l, r, t, b
#         self.same = same

#     def _padding(self, x):
#         if self.same:
#             ih, iw = x.size()[2:]
#             if ih % self.stride[0] == 0:
#                 ph = max(self.k[0] - self.stride[0], 0)
#             else:
#                 ph = max(self.k[0] - (ih % self.stride[0]), 0)
#             if iw % self.stride[1] == 0:
#                 pw = max(self.k[1] - self.stride[1], 0)
#             else:
#                 pw = max(self.k[1] - (iw % self.stride[1]), 0)
#             pl = pw // 2
#             pr = pw - pl
#             pt = ph // 2
#             pb = ph - pt
#             padding = (pl, pr, pt, pb)
#         else:
#             padding = self.padding
#         return padding

#     def forward(self, x):
#         # using existing pytorch functions and tensor ops so that we get autograd, 
#         # would likely be more efficient to implement from scratch at C/Cuda level
#         x = F.pad(x, self._padding(x), mode='reflect')
#         x = x.unfold(2, self.k[0], self.stride[0]).unfold(3, self.k[1], self.stride[1])
#         x = x.contiguous().view(x.size()[:4] + (-1,)).median(dim=-1)[0]
#         return x

# def get_uperleft_denominator(img, kernel):
#     # print("img",img.shape)
#     ker_f = convert_psf2otf(kernel, img.size()[-2:]) # discrete fourier transform of kernel
#     nsr = wiener_filter_para(img)
#     denominator = inv_fft_kernel_est(ker_f, nsr)#

#     # numerator = torch.rfft(img1, 3, onesided=False)
#     numerator = torch.view_as_real(torch.fft.fft2(img))
#     # print("numerator",numerator.shape)
#     # print("denominator",denominator.shape)
#     deblur = deconv(denominator, numerator)
#     return deblur

# # --------------------------------
# # --------------------------------
# def wiener_filter_para(_input_blur):
#     b,c,_,_ = _input_blur.shape
#     median_filter = MedianPool2d(kernel_size=3, padding=1)(_input_blur)
#     diff = median_filter - _input_blur
#     num = (diff.shape[2]**2)
#     mean_n = torch.sum(diff, (2, 3), keepdim=True)/num # .view(-1,1,1,1)
#     # print(mean_n.shape)
#     var_n = torch.sum((diff - mean_n) ** 2, (2,3), keepdim=True)/(num-1)
#     mean_input = torch.sum(_input_blur, (2,3), keepdim=True)/num
#     var_s2 = (torch.sum((_input_blur-mean_input)**2, (2, 3), keepdim=True)/(num-1))**(0.5)
#     NSR = var_n / var_s2 * 8.0 / 3.0 / 10.0
#     # NSR = NSR.view(-1,1,1,1)
#     return NSR

# # --------------------------------
# # --------------------------------
# def inv_fft_kernel_est(ker_f, NSR):
#     inv_denominator = ker_f[:, :, :, :, 0] * ker_f[:, :, :, :, 0] \
#                       + ker_f[:, :, :, :, 1] * ker_f[:, :, :, :, 1] + NSR

#     b,_,h,w = inv_denominator.shape
#     c = ker_f.shape[1]
#     inv_ker_f = torch.zeros((b,c,h,w,2)).to(ker_f.device)
#     # print("inv_ker_f",inv_ker_f.shape)
#     inv_ker_f[:, :, :, :, 0] = ker_f[:, :, :, :, 0] / inv_denominator
#     inv_ker_f[:, :, :, :, 1] = -ker_f[:, :, :, :, 1] / inv_denominator
#     return inv_ker_f

# # --------------------------------
# # --------------------------------
# def deconv(inv_ker_f, fft_input_blur):
#     # delement-wise multiplication.
#     deblur_f = torch.zeros_like(inv_ker_f)
#     deblur_f[:, :, :, :, 0] = inv_ker_f[:, :, :, :, 0] * fft_input_blur[:, :, :, :, 0] \
#                             - inv_ker_f[:, :, :, :, 1] * fft_input_blur[:, :, :, :, 1]
#     deblur_f[:, :, :, :, 1] = inv_ker_f[:, :, :, :, 0] * fft_input_blur[:, :, :, :, 1] \
#                             + inv_ker_f[:, :, :, :, 1] * fft_input_blur[:, :, :, :, 0]
#     deblur =  torch.fft.ifft2(torch.view_as_complex(deblur_f)).real
#     # deblur = torch.irfft(deblur_f, 3, onesided=False)
#     return deblur

# # --------------------------------
# # --------------------------------
# def convert_psf2otf(ker, size):
#     psf = torch.zeros(ker.shape[0],ker.shape[1],size[0],size[1]).to(ker.device)
#     # circularly shift
#     centre = ker.shape[2]//2 + 1
#     # 0:11    10:21
#     psf[:, :, :centre, :centre] = ker[:, :, (centre-1):, (centre-1):]
#     psf[:, :, :centre, -(centre-1):] = ker[:, :, (centre-1):, :(centre-1)]
#     psf[:, :, -(centre-1):, :centre] = ker[:, :, : (centre-1), (centre-1):]
#     psf[:, :, -(centre-1):, -(centre-1):] = ker[:, :, :(centre-1), :(centre-1)] 
#     # compute the otf
#     # otf = torch.fft.fftn(psf, dim=(-3, -2, -1))

#     otf = torch.view_as_real(torch.fft.fft2(psf))
#     # otf = torch.rfft(psf, 3, onesided=False)
#     return otf




# def postprocess(*images, rgb_range):
#     def _postprocess(img):
#         pixel_range = 255 / rgb_range
#         return img.mul(pixel_range).clamp(0, 255).round().div(pixel_range)

#     return [_postprocess(img) for img in images]

# def default_conv(in_channels, out_channels, kernel_size, bias=True):
#     return nn.Conv2d(
#         in_channels, out_channels, kernel_size,
#         padding=(kernel_size//2), bias=bias)

# class Conv(nn.Module):
#     def __init__(self , input_channels , n_feats , kernel_size , stride = 1 ,padding=0 , bias=True , bn = False , act=False ):
#         super(Conv , self).__init__()
#         m = []
#         m.append(nn.Conv2d(input_channels , n_feats , kernel_size , stride , padding , bias=bias))
#         if bn: m.append(nn.BatchNorm2d(n_feats))
#         if act:m.append(nn.ReLU(True))
#         self.body = nn.Sequential(*m)
#     def forward(self, input):
#         return self.body(input)

# class Deconv(nn.Module):
#     def __init__(self, input_channels, n_feats, kernel_size, stride=2, padding=0, output_padding=0 , bias=True, act=False):
#         super(Deconv, self).__init__()
#         m = []
#         m.append(nn.ConvTranspose2d(input_channels, n_feats, kernel_size, stride=stride, padding=padding,output_padding=output_padding, bias=bias))
#         if act: m.append(nn.ReLU(True))
#         self.body = nn.Sequential(*m)

#     def forward(self, input):
#         return self.body(input)

# class ResBlock(nn.Module):
#     def __init__(self, conv, n_feat, kernel_size, padding = 0 ,bias=True, bn=False, act=nn.ReLU(True), res_scale=1):
#         super(ResBlock, self).__init__()
#         m = []
#         for i in range(2):
#             m.append(conv(n_feat, n_feat, kernel_size, padding = padding , bias=bias))
#             if bn: m.append(nn.BatchNorm2d(n_feat))
#             if i == 0: m.append(act)

#         self.body = nn.Sequential(*m)
#         self.res_scale = res_scale

#     def forward(self, x):
#         res = self.body(x).mul(self.res_scale)
#         res += x

#         return res
# def make_model(args, parent=False):
#     return DWDN(args)
# @ARCH_REGISTRY.register()
# class DWDN(nn.Module):
#     def __init__(self , n_colors=3,psf_bases_num=10):
#         super().__init__()

#         n_resblock = 3
#         n_feats1 = 16
#         n_feats = 32
#         kernel_size = 5
#         self.psf_bases = np.load('psf_components.npy')
#         self.psf_bases = torch.from_numpy(self.psf_bases).cuda().reshape(psf_bases_num,21,21).unsqueeze(0)
#         self.kernel = nn.Parameter(self.psf_bases,requires_grad=True)
        
#         FeatureBlock = [Conv(n_colors*psf_bases_num, n_feats1, kernel_size, padding=2, act=True),
#                         ResBlock(Conv, n_feats1, kernel_size, padding=2),
#                         ResBlock(Conv, n_feats1, kernel_size, padding=2),
#                         ResBlock(Conv, n_feats1, kernel_size, padding=2)]

#         InBlock1 = [Conv(n_feats1, n_feats, kernel_size, padding=2, act=True),
#                    ResBlock(Conv, n_feats, kernel_size, padding=2),
#                    ResBlock(Conv, n_feats, kernel_size, padding=2),
#                    ResBlock(Conv, n_feats, kernel_size, padding=2)]
#         InBlock2 = [Conv(n_feats1 + n_feats, n_feats, kernel_size, padding=2, act=True),
#                    ResBlock(Conv, n_feats, kernel_size, padding=2),
#                    ResBlock(Conv, n_feats, kernel_size, padding=2),
#                    ResBlock(Conv, n_feats, kernel_size, padding=2)]

#         # encoder1
#         Encoder_first= [Conv(n_feats , n_feats*2 , kernel_size , padding = 2 ,stride=2 , act=True),
#                         ResBlock(Conv , n_feats*2 , kernel_size ,padding=2),
#                         ResBlock(Conv , n_feats*2 , kernel_size ,padding=2),
#                         ResBlock(Conv , n_feats*2 , kernel_size ,padding=2)]
#         # encoder2
#         Encoder_second = [Conv(n_feats*2 , n_feats*4 , kernel_size , padding=2 , stride=2 , act=True),
#                           ResBlock(Conv , n_feats*4 , kernel_size , padding=2),
#                           ResBlock(Conv , n_feats*4 , kernel_size , padding=2),
#                           ResBlock(Conv , n_feats*4 , kernel_size , padding=2)]
#         # decoder2
#         Decoder_second = [ResBlock(Conv , n_feats*4 , kernel_size , padding=2) for _ in range(n_resblock)]
#         Decoder_second.append(Deconv(n_feats*4 , n_feats*2 ,kernel_size=3 , padding=1 , output_padding=1 , act=True))
#         # decoder1
#         Decoder_first = [ResBlock(Conv , n_feats*2 , kernel_size , padding=2) for _ in range(n_resblock)]
#         Decoder_first.append(Deconv(n_feats*2 , n_feats , kernel_size=3 , padding=1, output_padding=1 , act=True))

#         OutBlock = [ResBlock(Conv , n_feats , kernel_size , padding=2) for _ in range(n_resblock)]

#         OutBlock2 = [Conv(n_feats , n_colors, kernel_size , padding=2)]

#         self.FeatureBlock = nn.Sequential(*FeatureBlock)
#         self.inBlock1 = nn.Sequential(*InBlock1)
#         self.inBlock2 = nn.Sequential(*InBlock2)
#         self.encoder_first = nn.Sequential(*Encoder_first)
#         self.encoder_second = nn.Sequential(*Encoder_second)
#         self.decoder_second = nn.Sequential(*Decoder_second)
#         self.decoder_first = nn.Sequential(*Decoder_first)
#         self.outBlock = nn.Sequential(*OutBlock)
#         self.outBlock2 = nn.Sequential(*OutBlock2)

#     def forward(self, input):
#         # kernel:(b,5,21,21)
#         input = input[:,:3,:,:]
#         # for jj in range(kernel.shape[0]):
#         #     kernel[jj:jj+1,:,:,:] = torch.div(kernel[jj:jj+1,:,:,:], torch.sum(kernel[jj:jj+1,:,:,:]))
#         ks = self.kernel.shape[2]
#         dim = (ks, ks, ks, ks)
#         input_pad = F.pad(input, dim, "replicate")
#         clean_inputs=[]
#         for i in range(input_pad.shape[1]):
#             clean_inputs.append(get_uperleft_denominator(input_pad[:,i:i+1,:,:], self.kernel)[:, :, ks:-ks, ks:-ks])
#         clean_inputs = torch.cat(clean_inputs, dim=1)
            
#         clear_features = self.FeatureBlock(clean_inputs)
#         # clear_features = torch.zeros(feature_out.size())
#         # ks = self.kernel.shape[2]
#         # dim = (ks, ks, ks, ks)
#         # first_scale_inblock_pad = F.pad(feature_out, dim, "replicate")
#         # clear_features =[]
#         # for i in range(first_scale_inblock_pad.shape[1]):
#         #     blur_feature_ch = first_scale_inblock_pad[:, i:i + 1, :, :]
#         #     # print(blur_feature_ch.shape) 
#         #     clear_feature_ch = get_uperleft_denominator(blur_feature_ch, self.kernel)
#         #     clear_features.append(clear_feature_ch[:, :, ks:-ks, ks:-ks])
#         #     # print("clear_feature_ch",clear_feature_ch.shape)

#         #     # clear_features[:, i:i + 1, :, :] = clear_feature_ch[:, :, ks:-ks, ks:-ks]
#         # clear_features = torch.cat(clear_features, dim=1)
#         self.n_levels = 2
#         self.scale = 0.5
#         output = []
#         for level in range(self.n_levels):
#             scale = self.scale ** (self.n_levels - level - 1)
#             n, c, h, w = input.shape
#             hi = int(round(h * scale))
#             wi = int(round(w * scale))
#             if level == 0:
#                 input_clear = F.interpolate(clear_features, (hi, wi), mode='bilinear')
#                 inp_all = input_clear
#                 first_scale_inblock = self.inBlock1(inp_all)
#             else:
#                 input_clear = F.interpolate(clear_features, (hi, wi), mode='bilinear')
#                 input_pred = F.interpolate(input_pre, (hi, wi), mode='bilinear')
#                 inp_all = torch.cat((input_clear, input_pred), 1)
#                 first_scale_inblock = self.inBlock2(inp_all)

#             first_scale_encoder_first = self.encoder_first(first_scale_inblock)
#             first_scale_encoder_second = self.encoder_second(first_scale_encoder_first)
#             first_scale_decoder_second = self.decoder_second(first_scale_encoder_second)
#             first_scale_decoder_first = self.decoder_first(first_scale_decoder_second+first_scale_encoder_first)
#             input_pre = self.outBlock(first_scale_decoder_first+first_scale_inblock)
#             # print("input_pre",input_pre.shape)
#             out = self.outBlock2(input_pre)
#             # print("out",out.shape)
#             output.append(out)
#             # print("output",output[0].shape)
#         output.reverse()
#         return output

# if __name__ == '__main__':
#     model = DWDN()
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model = model.to(device)
#     img = torch.randn(4,3,256,256).to(device)
#     # kernel=torch.randn(1,1,21,21).to(device)
#     kernel = np.load('psf_components.npy')
#     kernel = torch.from_numpy(kernel).to(device)
#     # # kernel = kernel.unsqueeze(1)
#     kernel = kernel.reshape(1,10,21,21)
#     # print(kernel.shape)
#     output = model(img)
#     print(output[0].shape)
#     for i in range(len(output)):
#         print(output[i].shape)
    
#     # psf = torch.ones(3, 3)  # Example PSF (you can replace this with your actual PSF)
#     # out_size = (12, 12)  # Desired output size
#     # otf = psf2otf(psf, out_size)
#     # pad_img = zero_pad(np.ones((3,3)), (9,9), position='center')
#     # print(pad_img)
#     # print(otf)