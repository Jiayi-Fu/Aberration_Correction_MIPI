
import numpy as np
import cv2
import torch
def addnoise_singleChannel(y, a, b):
    # 将输入的 HWC 格式转换为 numpy 数组
    y = np.asarray(y)  # 确保输入是 numpy 数组
    sigma2 = a*y+b
    sigma2 = np.array(sigma2)
    sigma2[sigma2<0] = 0
    z = y + np.sqrt(sigma2)*np.random.randn(y.shape[0],y.shape[1])
    z = np.clip(z,0.0,1.)
    return z


def addnoise(img_aberration, noise):
    noiseR = noise['noiseR']
    noiseG = noise['noiseG']
    noiseB = noise['noiseB']

    img_aberration_noise = np.zeros_like(img_aberration,dtype=np.float32)
    
    img_aberration_noise[:,:,0] = addnoise_singleChannel(img_aberration[:,:,0], noiseR['fitparams'][0,0][0,0],noiseR['fitparams'][0,0][0,1])
    img_aberration_noise[:,:,1] = addnoise_singleChannel(img_aberration[:,:,1], noiseG['fitparams'][0,0][0,0],noiseG['fitparams'][0,0][0,1])
    img_aberration_noise[:,:,2] = addnoise_singleChannel(img_aberration[:,:,2], noiseB['fitparams'][0,0][0,0],noiseB['fitparams'][0,0][0,1])

    return img_aberration_noise



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

def make_mosaic(im,pattern):
    w,h,c=im.shape
    R=np.zeros((w,h))
    GR = np.zeros((w, h))
    GB = np.zeros((w, h))
    B = np.zeros((w, h))
    image_data=im
    #将对应位置的元素取出来,因为懒所以没有用效率最高的方法,大家可以自己去实现
    if (pattern == "RGGB"):
        R[::2, ::2]= image_data[::2, ::2, 0]
        GR[::2, 1::2] = image_data[::2, 1::2, 1]
        GB[1::2, ::2] = image_data[1::2, ::2, 1]
        B[1::2, 1::2]= image_data[1::2, 1::2, 2]
    elif (pattern == "GRBG"):
        GR[::2, ::2] = image_data[::2, ::2, 1]
        R[::2, 1::2] = image_data[::2, 1::2, 0]
        B[1::2, ::2] = image_data[1::2, ::2, 2]
        GB[1::2, 1::2] = image_data[1::2, 1::2, 1]
    elif (pattern == "GBRG"):
        GB[::2, ::2] = image_data[::2, ::2, 1]
        B[::2, 1::2] = image_data[::2, 1::2, 2]
        R[1::2, ::2] = image_data[1::2, ::2, 0]
        GR[1::2, 1::2] = image_data[1::2, 1::2, 1]
    elif (pattern == "BGGR"):
        B[::2, ::2] = image_data[::2, ::2, 2]
        GB[::2, 1::2] = image_data[::2, 1::2, 1]
        GR[1::2, ::2] = image_data[1::2, ::2, 1]
        R[1::2, 1::2] = image_data[1::2, 1::2, 0]
    else:
        print("pattern must be one of :  RGGB, GBRG, GBRG, or BGGR")
        return
    result_image=R+GR+GB+B
    return result_image

    # X(:, 1: 2:end, 1)=0;
    # X(2: 2:end, 2: 2:end, 1)=0;
    # X(2: 2:end, 1: 2:end, 2)=0;
    # X(1: 2:end, 2: 2:end, 2)=0;
    # X(1: 2:end, 1: 2:end, 3)=0;
    # X(:, 2: 2:end, 3)=0;

def chromatic_aberration(rgb, translations, scaling):
    """
    Introduces chromatic aberration effects.

    :param rgb: 3xHxW input RGB image
    :param translations: 3x2 translation tensor (tx, ty) for each of R, G, B
    :param scaling: [sr, sg, sb] scaling factor for each of R, G, B
    :return: 3xHxW output tensor
    """

    assert rgb.dim() == 3, "input tensor has invalid size {}".format(rgb.size())
    assert rgb.size(0) == 3, "input tensor has invalid size {}".format(rgb.size())

    transformations = torch.zeros(3, 2, 3)
    transformations[:,0,0] = scaling
    transformations[:,1,1] = scaling
    transformations[:,0:2,2] = translations

    grid = torch.nn.functional.affine_grid(transformations.to(rgb.device), (3,1,rgb.size(1),rgb.size(2)), align_corners=False)

    sampled = torch.nn.functional.grid_sample(rgb.unsqueeze(1), grid,
        mode='bilinear', padding_mode='reflection',
        align_corners=False
    )

    return sampled[:,0]

def mosaic(image, bayer_pattern):
    r = image[::2, ::2, 0:1]
    gr = image[::2, 1::2, 1:2]
    gb = image[1::2, ::2, 1:2]
    b = image[1::2, 1::2, 2:3]

    if bayer_pattern == 'rgbg':
        out = np.concatenate([r, gr, b, gb], axis=-1)
    elif bayer_pattern == 'rggb':
        out = np.concatenate([r, gr, gb, b], axis=-1)
    elif bayer_pattern == 'grbg':
        out = np.concatenate([gr, r, b, gb], axis=-1)
    else:
        raise ValueError(f"Unknown output type: {bayer_pattern}")

    return out

def pixel_shuffle(x, upscale_factor):
    H, W, C = x.shape
    x = x.transpose(2, 0, 1)

    assert C % (upscale_factor ** 2) == 0

    x = x.reshape(upscale_factor, upscale_factor, H, W)

    x = x.transpose(2 , 0, 3, 1)
    x = x.reshape(H * upscale_factor, W * upscale_factor)

    return x

def pixel_unshuffle(x, downscale_factor):
    H, W = x.shape
    assert H % downscale_factor == 0 and W % downscale_factor == 0

    new_H = H // downscale_factor
    new_W = W // downscale_factor

    x = x.reshape(new_H, downscale_factor, new_W, downscale_factor)
    x = x.transpose(0, 2, 1, 3)
    return x.reshape(new_H, new_W, downscale_factor * downscale_factor)


def demosaic(bayer_images, bayer_pattern='rggb'):
    """Bilinearly demosaics a batch of RGGB Bayer images."""

    def bilinear_interpolate(x, shape):
        return cv2.resize(x, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)

    H, W, C = bayer_images.shape

    assert C == 4  # 确保输入是 RGGB
    shape = [H * 2, W * 2]


    red = bayer_images[:, :,0:1]
    green_red = bayer_images[:, :, 1:2]
    if bayer_pattern == 'rggb':
        green_blue = bayer_images[:, :, 2:3]
        blue = bayer_images[:, :, 3:4]
    elif bayer_pattern == 'rgbg':
        blue = bayer_images[:, :, 2:3]
        green_blue = bayer_images[:, :, 3:4]

    red = bilinear_interpolate(red, shape)

    green_red = cv2.flip(green_red, 1)
    green_red = bilinear_interpolate(green_red, shape)
    green_red = cv2.flip(green_red, 1)

    green_red = pixel_unshuffle(green_red, 2)

    green_blue = cv2.flip(green_blue, 0)
    green_blue = bilinear_interpolate(green_blue, shape)
    green_blue = cv2.flip(green_blue, 0)
    green_blue = pixel_unshuffle(green_blue, 2)


    green_at_red = (green_red[:,:,0] + green_blue[:,:,0]) / 2
    green_at_green_red = green_red[:, :,1]
    green_at_green_blue = green_blue[:, :,2]
    green_at_blue = (green_red[:, :,3] + green_blue[:, :,3]) / 2


    green_planes = [
        green_at_red, green_at_green_red, green_at_green_blue, green_at_blue
    ]

    green = pixel_shuffle(np.stack(green_planes, axis=2), 2)

    blue = np.flip(np.flip(blue, axis=0), axis=1)
    blue = bilinear_interpolate(blue, shape)
    blue = np.flip(np.flip(blue, axis=0), axis=1)


    rgb_images = np.stack([red, green, blue], axis=-1)

    # if T is not None:
    #     rgb_images = rgb_images.reshape(B, T, 3, H * 2, W * 2)

    return rgb_images


def bayer_pattern_unify(raw, bayer_pattern):
    if bayer_pattern == 'RGGB':
        raw = raw[::-1, ::-1]
    elif bayer_pattern == 'GRBG':
        raw = raw[::-1, :]
    elif bayer_pattern == 'GBRG':
        raw = raw[:, ::-1]
    elif bayer_pattern == 'BGGR':
        pass
    return raw


def demosaic(raw):
    raw = (raw * 65535).astype(np.uint16)
    bgr = cv2.cvtColor(raw, cv2.COLOR_BAYER_BG2BGR_EA)
    bgr = bgr.astype(np.float32) / 65535.
    # bgr = bgr.astype(np.float32) / 1023.
    return bgr

def pack_raw_bayer(raw: np.ndarray, raw_pattern: np.ndarray):
    #pack Bayer image to 4 channels RGBG
    R = np.where(raw_pattern==0)
    G1 = np.where(raw_pattern==1)
    B = np.where(raw_pattern==2)
    G2 = np.where(raw_pattern==3)

    raw = raw.astype(np.uint16)
    out = np.stack((raw[R[0][0]::2,  R[1][0]::2], #RGGB
                    raw[G1[0][0]::2, G1[1][0]::2],
                    raw[G2[0][0]::2, G2[1][0]::2],
                    raw[B[0][0]::2,  B[1][0]::2]),axis=0)

    return out
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
