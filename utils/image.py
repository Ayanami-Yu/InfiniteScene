#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import torch


def mse(img1, img2):
    return (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)


def psnr(img1, img2):
    mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))


def crop_image(img, H=320, W=512):
    w_in, h_in = img.size
    # crop to a square to resize proportionally
    max_len = max(H, W)
    if w_in > h_in:
        img = img.crop((int(w_in / 2 - h_in / 2), 0, int(w_in / 2 + h_in / 2), h_in))
    else:  # w <= h
        img = img.crop((0, int(h_in / 2 - w_in / 2), w_in, int(h_in / 2 + w_in / 2)))
    img = img.resize(max_len, max_len)
    # crop to target height and width
    if H < max_len:
        img = img.crop((0, int(max_len / 2 - H / 2), 0, int(max_len / 2 + H / 2)))
    elif W < max_len:
        img = img.crop((int(max_len / 2 - W / 2), 0, int(max_len / 2 + W / 2), 0))
    return img
