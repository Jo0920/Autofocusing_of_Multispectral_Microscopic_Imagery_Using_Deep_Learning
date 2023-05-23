import os

import numpy as np
import torch
import torchvision.transforms as transforms
from box import Box
from torch.autograd import Variable
import torch
import torch.cuda
from torch.autograd import Variable
from skimage.color import rgb2ycbcr, ycbcr2rgb

def get_cfg():
    current_path = os.path.abspath(".")
    yaml_path = os.path.join(current_path, "cfg.yaml")
    conf = Box.from_yaml(filename=yaml_path)
    return conf

cfg = get_cfg()

def img2patch(img):
    y = 0
    x = 0
    res = []
    cfg = get_cfg()
    while y < cfg.img_height + 1 - cfg.patch_height:
        while x < cfg.img_width + 1 - cfg.patch_width:
            cropped = img[0].crop((x, y, x + cfg.patch_width, y + cfg.patch_height))
            tensor_cropped = transforms.ToTensor()(cropped)
            res.append(tensor_cropped)
            x += cfg.patch_width
        x = 0
        y += cfg.patch_height
    return res


def patch2img(generated_list):
    row_list = []
    column_list = []
    count = 0
    cfg = get_cfg()
    for g_img in generated_list:
        row_list.append(g_img)
        count += 1
        if (count % (cfg.img_width // cfg.patch_width)) == 0:
            _row = torch.cat(row_list, -1)
            column_list.append(_row)
            row_list = []
            count = 0
    final_img = torch.cat(column_list, -2)
    return final_img


def ms_img2patch_2d(imgs, ispredict, spectrum_num=cfg.spectrum_num):
    """
    get tensor image list
    output: 3D tensor
    [c, h, w]
    """
    y = 0
    x = 0
    res = []
    cfg = get_cfg()
    while y < cfg.img_height + 1 - cfg.patch_height:
        while x < cfg.img_width + 1 - cfg.patch_width:
            cropped = imgs[0].crop((x, y, x + cfg.patch_width, y + cfg.patch_height))
            if (len(np.shape(cropped))) < 3:
                cropped = np.array(cropped)[:, :, None]
            for i in range(1, spectrum_num):
                new_cropped = imgs[i].crop((x, y, x + cfg.patch_width, y + cfg.patch_height))
                if (len(np.shape(new_cropped))) < 3:
                    new_cropped = np.array(new_cropped)[:, :, None]
                cropped = np.append(cropped, new_cropped, axis=2)
            tensor_cropped = transforms.ToTensor()(cropped)
            res.append(tensor_cropped)
            if ispredict:
                x += cfg.patch_width
            else:
                x += cfg.patch_width // 2
        x = 0
        if ispredict:
            y += cfg.patch_height
        else:
            y += cfg.patch_height // 2
    return res


def ms_img2patch_3d(imgs, ispredict):
    """
    get tensor image list
    output: 4D tensor
    [c, spectrum_num h, w]
    """
    y = 0
    x = 0
    res = []
    cfg = get_cfg()
    while y < cfg.img_height + 1 - cfg.patch_height:
        while x < cfg.img_width + 1 - cfg.patch_width:
            cropped = (
                transforms.ToTensor()(imgs[0].crop((x, y, x + cfg.patch_width, y + cfg.patch_height)))).unsqueeze(1)
            for i in range(1, cfg.spectrum_num):
                cropped_new = (
                    transforms.ToTensor()(imgs[i].crop((x, y, x + cfg.patch_width, y + cfg.patch_height)))).unsqueeze(1)
                cropped = torch.cat((cropped, cropped_new), 1)
            res.append(cropped)
            if ispredict:
                x += cfg.patch_width
            else:
                x += cfg.patch_width // 2
        x = 0
        if ispredict:
            y += cfg.patch_height
        else:
            y += cfg.patch_height // 2
    return res




'''
def _convert(input_, type_):
    return {
        'float': input_.float(),
        'double': input_.double(),
    }.get(type_, input_)


def _generic_transform_sk_4d(transform, in_type='', out_type=''):
    def apply_transform(input_):
        to_squeeze = (input_.dim() == 3)
        device = input_.device
        input_ = input_.cpu()
        input_ = _convert(input_, in_type)

        if to_squeeze:
            input_ = input_.unsqueeze(0)

        input_ = input_.permute(0, 2, 3, 1).numpy()
        transformed = transform(input_)
        output = torch.from_numpy(transformed).float().permute(0, 3, 1, 2)
        if to_squeeze:
            output = output.squeeze(0)
        output = _convert(output, out_type)
        return output.to(device)

    return apply_transform


# --- YCbCr ---
rgb_to_ycbcr = _generic_transform_sk_4d(rgb2ycbcr)
ycbcr_to_rgb = _generic_transform_sk_4d(ycbcr2rgb, in_type='double', out_type='float')
'''


def ycbcr_to_rgb(image: torch.Tensor) -> torch.Tensor:
    r"""Convert an YCbCr image to RGB.

    The image data is assumed to be in the range of (0, 1).

    Args:
        image: YCbCr Image to be converted to RGB with shape :math:`(*, 3, H, W)`.

    Returns:
        RGB version of the image with shape :math:`(*, 3, H, W)`.

    Examples:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> output = ycbcr_to_rgb(input)  # 2x3x4x5
    """
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(image)}")

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError(f"Input size must have a shape of (*, 3, H, W). Got {image.shape}")

    y: torch.Tensor = image[..., 0, :, :]
    cb: torch.Tensor = image[..., 1, :, :]
    cr: torch.Tensor = image[..., 2, :, :]
    '''

    Y  = R *  0.29900 + G *  0.58700 + B *  0.11400
    Cb = R * -0.16874 + G * -0.33126 + B *  0.50000 + 128
    Cr = R *  0.50000 + G * -0.41869 + B * -0.08131 + 128
    R  = Y +                       + (Cr - 128) *  1.40200
    G  = Y + (Cb - 128) * -0.34414 + (Cr - 128) * -0.71414
    B  = Y + (Cb - 128) *  1.77200
    '''

    delta: float = 0.5
    cb_shifted: torch.Tensor = cb - delta
    cr_shifted: torch.Tensor = cr - delta

    r: torch.Tensor = y + 1.40200 * cr_shifted
    g: torch.Tensor = y - 0.71414 * cr_shifted - 0.34414 * cb_shifted
    b: torch.Tensor = y + 1.77200 * cb_shifted

    zero = torch.zeros(np.shape(r))
    zero = zero.cuda()
    r = torch.max(r, zero)
    g = torch.max(g, zero)
    b = torch.max(b, zero)
    '''
    delta: float = 0.5
    r: torch.Tensor = y + 1.403 * cr_shifted
    g: torch.Tensor = y - 0.714 * cr_shifted - 0.344 * cb_shifted
    b: torch.Tensor = y + 1.773 * cb_shifted
    '''
    return torch.stack([r, g, b], -3)
