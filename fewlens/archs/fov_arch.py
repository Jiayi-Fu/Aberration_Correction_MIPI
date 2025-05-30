import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import numpy as np
try:
    from fewlens.dcn.deform_conv import ModulatedDeformConvPack2 as DCN
except ImportError:
    raise ImportError('Failed to import DCNv2 module.')
from fewlens.utils.registry import ARCH_REGISTRY
#==============================================================================#
class ResBlock(nn.Module):

    def __init__(self, input_channel=32, output_channel=32):
        super().__init__()
        self.in_channel = input_channel
        self.out_channel = output_channel
        if self.in_channel != self.out_channel:
            self.conv0 = nn.Conv2d(input_channel, output_channel, 1, 1)
        self.conv1 = nn.Conv2d(output_channel, output_channel, 3, 1, 1)
        self.conv2 = nn.Conv2d(output_channel, output_channel, 3, 1, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.initialize_weights()

    def forward(self, x):
        if self.in_channel != self.out_channel:
            x = self.conv0(x)
        conv1 = self.lrelu(self.conv1(x))
        conv2 = self.conv2(conv1)
        out = x + conv2
        return out
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

class RSABlock(nn.Module):

    def __init__(self, input_channel=32, output_channel=32, offset_channel=32):
        super().__init__()
        self.in_channel = input_channel
        self.out_channel = output_channel
        if self.in_channel != self.out_channel:
            self.conv0 = nn.Conv2d(input_channel, output_channel, 1, 1)
        self.dcnpack = DCN(output_channel, output_channel, 3, stride=1, padding=1, dilation=1, deformable_groups=8,
                            extra_offset_mask=True, offset_in_channel=offset_channel)
        self.conv1 = nn.Conv2d(output_channel, output_channel, 3, 1, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.initialize_weights()

    def forward(self, x, offset):
        if self.in_channel != self.out_channel:
            x = self.conv0(x)
        fea = self.lrelu(self.dcnpack([x, offset]))
        out = self.conv1(fea) + x
        return out
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

class OffsetBlock(nn.Module):

    def __init__(self, input_channel=32, offset_channel=32, last_offset=False):
        super().__init__()
        self.offset_conv1 = nn.Conv2d(input_channel, offset_channel, 3, 1, 1)  # concat for diff
        if last_offset:
            self.offset_conv2 = nn.Conv2d(offset_channel*2, offset_channel, 3, 1, 1)  # concat for offset
        self.offset_conv3 = nn.Conv2d(offset_channel, offset_channel, 3, 1, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.initialize_weights()

    def forward(self, x, last_offset=None):
        offset = self.lrelu(self.offset_conv1(x))
        if last_offset is not None:
            last_offset = F.interpolate(last_offset, scale_factor=2, mode='bilinear', align_corners=False)
            offset = self.lrelu(self.offset_conv2(torch.cat([offset, last_offset * 2], dim=1)))
        offset = self.lrelu(self.offset_conv3(offset))
        return offset
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

class FOVBlock(nn.Module):

    def __init__(self, input_channel=5, output_channel=32, nf=32):
        super().__init__()
        # spatial attention
        self.sAtt_fov_1 = nn.Conv2d(input_channel-3, nf, 3, 1, 1, bias=True)
        self.sAtt_fov_2 = nn.Conv2d(nf, nf, 1, 1, bias=True)

        self.sAtt_img_1 = nn.Conv2d(3, nf, 3, 1, 1, bias=True)
        self.sAtt_img_2 = nn.Conv2d(nf, nf, 1, 1, bias=True)

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, imgwzfov):
        B, N, H, W = imgwzfov.size()  # img with fov information
        img = imgwzfov[:, 0:3, :, :]
        fov = imgwzfov[:, 3:N, :, :]

        att_img = self.lrelu(self.sAtt_img_1(img))
        att_fov = self.lrelu(self.sAtt_fov_1(fov))
        att_img = self.lrelu(self.sAtt_img_2(att_img))
        att_fov = self.lrelu(self.sAtt_fov_2(att_fov))

        att_fov = torch.sigmoid(att_fov)

        att_fea = att_img * att_fov + att_img
        return att_fea
    

class KernelConv(nn.Module):
    """
    the class of kernel prediction
    """
    def __init__(self, kernel_size=[5]):
        super(KernelConv, self).__init__()
        self.kernel_size = sorted(kernel_size)

    def _convert_dict(self, core, batch_size, color, height, width):
        """
        make sure the core to be a dict, generally, only one kind of kernel size is suitable for the func.
        :param core: shape: batch_size*(N*K*K)*height*width
        :return: core_out, a dict
        """
        core_out = {}
        core = core.view(batch_size, color, -1, height, width)
        core_out[self.kernel_size[0]] = core[:, 0:self.kernel_size[0]**2, ...]
        return core_out

    def forward(self, inputs, core, rate=1):
        """
        compute the pred image according to the input and core
        :param inputs: [batch_size, color, height, width]
        :param core: [batch_size, dict(kernel), color, height, width]
        """
        img_stack = []
        pred_img = []
        batch_size, color, height, width = inputs.size()
        core = self._convert_dict(core, batch_size, color, height, width)

        K = self.kernel_size[0]
        padding_num = (K//2) * rate
        inputs_pad = F.pad(inputs, [padding_num, padding_num, padding_num, padding_num])
        for i in range(0, K):
            for j in range(0, K):
                img_stack.append(inputs_pad[..., i*rate:i*rate + height, j*rate:j*rate + width])
        img_stack = torch.stack(img_stack, dim=2)
        pred_img.append(torch.sum(
            core[K].mul(img_stack), dim=2, keepdim=False
        ))
        pred_img = torch.stack(pred_img, dim=0)
        pred_img = pred_img.squeeze(0)
        return pred_img

@ARCH_REGISTRY.register()
class FOVKPN(nn.Module):

    def __init__(self, input_channel=5, output_channel=3, n_channel=32, offset_channel=32,
                 fov_att=False, kernel_size=[5], color=True):
        super().__init__()
        output_kernel_channel = (3 if color else 1) * np.sum(np.array(kernel_size) ** 2)

        self.fovblock = FOVBlock(input_channel, n_channel, n_channel)
        self.res1 = ResBlock(n_channel, n_channel)
        self.down1 = nn.Conv2d(n_channel, n_channel*2, 2, 2)
        self.res2 = ResBlock(n_channel*2, n_channel*2)
        self.down2 = nn.Conv2d(n_channel*2, n_channel*4, 2, 2)
        self.res3 = ResBlock(n_channel*4, n_channel*4)
        self.down3 = nn.Conv2d(n_channel*4, n_channel*8, 2, 2)
        self.res4 = ResBlock(n_channel*8, n_channel*8)

        self.offset4 = OffsetBlock(n_channel*8, offset_channel, False)
        self.dres4 = RSABlock(n_channel*8, n_channel*8, offset_channel)

        self.up3 = nn.ConvTranspose2d(n_channel*8, n_channel*4, 2, 2)
        self.dconv3_1 = nn.Conv2d(n_channel*8, n_channel*4, 1, 1)
        self.offset3 = OffsetBlock(n_channel*4, offset_channel, True)
        self.dres3 = RSABlock(n_channel*4, n_channel*4, offset_channel)

        self.up2 = nn.ConvTranspose2d(n_channel*4, n_channel*2, 2, 2)
        self.dconv2_1 = nn.Conv2d(n_channel*4, n_channel*2, 1, 1)
        self.offset2 = OffsetBlock(n_channel*2, offset_channel, True)
        self.dres2 = RSABlock(n_channel*2, n_channel*2, offset_channel)

        self.up1 = nn.ConvTranspose2d(n_channel*2, n_channel, 2, 2)
        self.dconv1_1 = nn.Conv2d(n_channel*2, n_channel, 1, 1)
        self.offset1 = OffsetBlock(n_channel, offset_channel, True)
        self.dres1 = RSABlock(n_channel, n_channel, offset_channel)

        self.outc = nn.Conv2d(n_channel, output_kernel_channel, kernel_size=1, stride=1, padding=0)

        self.kernel_pred = KernelConv(kernel_size)

        self.conv_final = nn.Conv2d(in_channels=12, out_channels=3, kernel_size=3, stride=1, padding=1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        
        self.apply(self.initialize_weights)

    def forward(self, x):
        att_fov = self.fovblock(x)
        conv1 = self.res1(att_fov)
        pool1 = self.lrelu(self.down1(conv1))
        conv2 = self.res2(pool1)
        pool2 = self.lrelu(self.down2(conv2))
        conv3 = self.res3(pool2)
        pool3 = self.lrelu(self.down3(conv3))
        conv4 = self.res4(pool3)

        L4_offset = self.offset4(conv4, None)
        dconv4 = self.dres4(conv4, L4_offset)

        up3 = torch.cat([self.up3(dconv4), conv3], 1)
        up3 = self.dconv3_1(up3)
        L3_offset = self.offset3(up3, L4_offset)
        dconv3 = self.dres3(up3, L3_offset)

        up2 = torch.cat([self.up2(dconv3), conv2], 1)
        up2 = self.dconv2_1(up2)
        L2_offset = self.offset2(up2, L3_offset)
        dconv2 = self.dres2(up2, L2_offset)

        up1 = torch.cat([self.up1(dconv2), conv1], 1)
        up1 = self.dconv1_1(up1)
        L1_offset = self.offset1(up1, L2_offset)
        dconv1 = self.dres1(up1, L1_offset)

        core = self.outc(dconv1)

        pred1 = self.kernel_pred(x[:, 0:3, :, :], core, rate=1)
        pred2 = self.kernel_pred(x[:, 0:3, :, :], core, rate=2)
        pred3 = self.kernel_pred(x[:, 0:3, :, :], core, rate=3)
        pred4 = self.kernel_pred(x[:, 0:3, :, :], core, rate=4)

        pred_cat = torch.cat([torch.cat([torch.cat([pred1, pred2], dim=1), pred3], dim=1), pred4], dim=1)

        out = self.conv_final(pred_cat)
        return out
    
   
    def initialize_weights(self,m):

        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            #torch.nn.init.xavier_normal_(m.weight.data)
            torch.nn.init.xavier_uniform_(m.weight.data)
            #torch.nn.init.kaiming_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            torch.nn.init.normal_(m.weight.data, 0, 0.01)
            m.bias.data.zero_()
