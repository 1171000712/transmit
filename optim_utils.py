import torch
from datasets import load_dataset
from typing import Any, Mapping
import json
import numpy as np
import os
from statistics import mean, stdev
from image_utils import *
import copy


def read_json(filename: str) -> Mapping[str, Any]:
    """Returns a Python dict representation of JSON object at input file."""
    with open(filename) as fp:
        return json.load(fp)


def get_dataset(args):
    if 'laion' in args.dataset_path:
        dataset = load_dataset(args.dataset)['train']
        prompt_key = 'TEXT'
    elif 'coco' in args.dataset_path:
        with open('fid_outputs/coco/meta_data.json') as f:
            dataset = json.load(f)
            dataset = dataset['annotations']
            prompt_key = 'caption'
    else:
        dataset = load_dataset(args.dataset_path)['train']
        prompt_key = 'Prompt'
    return dataset, prompt_key

'''
def save_metrics(args, tpr_detection, tpr_traceability, acc, clip_scores):
    names = {
        'jpeg_ratio': "Jpeg.txt",
        'random_crop_ratio': "RandomCrop.txt",
        'random_drop_ratio': "RandomDrop.txt",
        'gaussian_blur_r': "GauBlur.txt",
        'gaussian_std': "GauNoise.txt",
        'median_blur_k': "MedBlur.txt",
        'resize_ratio': "Resize.txt",
        'sp_prob': "SPNoise.txt",
        'brightness_factor': "Color_Jitter.txt"
    }
    filename = "Identity.txt"
    for option, name in names.items():
        if getattr(args, option) is not None:
            filename = name

    if args.reference_model is not None:
        with open(args.output_path + filename, "a") as file:
            file.write('tpr_detection:' + str(tpr_detection / args.num) + '      ' +
                       'tpr_traceability:' + str(tpr_traceability / args.num) + '      ' +
                       'mean_acc:' + str(mean(acc)) + '      ' + 'std_acc:' + str(stdev(acc)) + '      ' +
                       'mean_clip_score:' + str(mean(clip_scores)) + '      ' + 'std_clip_score:' + str(stdev(clip_scores)) + '      ' +
                       '\n')

    else:
        with open(args.output_path + filename, "a") as file:
            file.write('tpr_detection:' + str(tpr_detection / args.num) + '      ' +
                       'tpr_traceability:' + str(tpr_traceability / args.num) + '      ' +
                       'mean_acc:' + str(mean(acc)) + '      ' + 'std_acc:' + str(stdev(acc)) + '      ' +
                       '\n')

    return
'''
def save_metrics(args, tpr_detection, tpr_traceability, acc, clip_scores):
    names = {
        'jpeg_ratio': "Jpeg.txt",
        'random_crop_ratio': "RandomCrop.txt",
        'random_drop_ratio': "RandomDrop.txt",
        'gaussian_blur_r': "GauBlur.txt",
        'gaussian_std': "GauNoise.txt",
        'median_blur_k': "MedBlur.txt",
        'resize_ratio': "Resize.txt",
        'sp_prob': "SPNoise.txt",
        'brightness_factor': "Color_Jitter.txt"
    }

    filename = "Identity.txt"
    for option, name in names.items():
        if getattr(args, option) is not None:
            filename = name

    if args.reference_model is not None:
        with open(args.output_path + args.filename+'.txt', "a") as file:
            file.write(filename[:-4]+':   '+'tpr_detection:' + str(tpr_detection / args.num) + '      ' +
                       'tpr_traceability:' + str(tpr_traceability / args.num) + '      ' +
                       'mean_acc:' + str(mean(acc)) + '      ' + 'std_acc:' + str(stdev(acc)) + '      ' +
                       'mean_clip_score:' + str(mean(clip_scores)) + '      ' + 'std_clip_score:' + str(stdev(clip_scores)) + '      ' +
                       '\n')

    else:
        with open(args.output_path + args.filename+'.txt', "a") as file:
            file.write(filename[:-4]+':   '+'tpr_detection:' + str(tpr_detection / args.num) + '      ' +
                       'tpr_traceability:' + str(tpr_traceability / args.num) + '      ' +
                       'mean_acc:' + str(mean(acc)) + '      ' + 'std_acc:' + str(stdev(acc)) + '      ' +
                       '\n')

    return

def circle_mask(size=64, r=10, x_offset=0, y_offset=0):
    # reference: https://stackoverflow.com/questions/69687798/generating-a-soft-circluar-mask-using-numpy-python-3
    x0 = y0 = size // 2
    x0 += x_offset
    y0 += y_offset
    y, x = np.ogrid[:size, :size]
    y = y[::-1]

    return ((x - x0)**2 + (y-y0)**2)<= r**2


def get_watermarking_mask(init_latents_w, args, device):
    watermarking_mask = torch.zeros(init_latents_w.shape, dtype=torch.bool).to(device)

    if args.w_mask_shape == 'circle':
        np_mask = circle_mask(init_latents_w.shape[-1], r=args.w_radius)
        torch_mask = torch.tensor(np_mask).to(device)

        if args.w_channel == -1:
            # all channels
            watermarking_mask[:, :] = torch_mask
        else:
            watermarking_mask[:, args.w_channel] = torch_mask
    elif args.w_mask_shape == 'square':
        anchor_p = init_latents_w.shape[-1] // 2
        if args.w_channel == -1:
            # all channels
            watermarking_mask[:, :, anchor_p-args.w_radius:anchor_p+args.w_radius, anchor_p-args.w_radius:anchor_p+args.w_radius] = True
        else:
            watermarking_mask[:, args.w_channel, anchor_p-args.w_radius:anchor_p+args.w_radius, anchor_p-args.w_radius:anchor_p+args.w_radius] = True
    elif args.w_mask_shape == 'no':
        pass
    else:
        raise NotImplementedError(f'w_mask_shape: {args.w_mask_shape}')

    return watermarking_mask


def get_watermarking_pattern(pipe, args, device, shape=None):
    set_random_seed(args.w_seed)
    if shape is not None:
        gt_init = torch.randn(*shape, device=device)
    else:
        gt_init = pipe.get_random_latents()

    if 'seed_ring' in args.w_pattern:
        gt_patch = gt_init

        gt_patch_tmp = copy.deepcopy(gt_patch)
        for i in range(args.w_radius, 0, -1):
            tmp_mask = circle_mask(gt_init.shape[-1], r=i)
            tmp_mask = torch.tensor(tmp_mask).to(device)
            
            for j in range(gt_patch.shape[1]):
                gt_patch[:, j, tmp_mask] = gt_patch_tmp[0, j, 0, i].item()
    elif 'seed_zeros' in args.w_pattern:
        gt_patch = gt_init * 0
    elif 'seed_rand' in args.w_pattern:
        gt_patch = gt_init
    elif 'rand' in args.w_pattern:
        gt_patch = torch.fft.fftshift(torch.fft.fft2(gt_init), dim=(-1, -2))
        gt_patch[:] = gt_patch[0]
    elif 'zeros' in args.w_pattern:
        gt_patch = torch.fft.fftshift(torch.fft.fft2(gt_init), dim=(-1, -2)) * 0
    elif 'const' in args.w_pattern:
        gt_patch = torch.fft.fftshift(torch.fft.fft2(gt_init), dim=(-1, -2)) * 0
        gt_patch += args.w_pattern_const
    elif 'ring' in args.w_pattern:
        gt_patch = torch.fft.fftshift(torch.fft.fft2(gt_init), dim=(-1, -2))

        gt_patch_tmp = copy.deepcopy(gt_patch)
        for i in range(args.w_radius, 0, -1):
            tmp_mask = circle_mask(gt_init.shape[-1], r=i)
            tmp_mask = torch.tensor(tmp_mask).to(device)
            
            for j in range(gt_patch.shape[1]):
                gt_patch[:, j, tmp_mask] = gt_patch_tmp[0, j, 0, i].item()

    return gt_patch


def inject_watermark(init_latents_w, watermarking_mask, gt_patch, args):
    init_latents_w_fft = torch.fft.fftshift(torch.fft.fft2(init_latents_w), dim=(-1, -2))
    if args.w_injection == 'complex':
        init_latents_w_fft[watermarking_mask] = gt_patch[watermarking_mask].clone()
    elif args.w_injection == 'seed':
        init_latents_w[watermarking_mask] = gt_patch[watermarking_mask].clone()
        return init_latents_w
    else:
        NotImplementedError(f'w_injection: {args.w_injection}')

    init_latents_w = torch.fft.ifft2(torch.fft.ifftshift(init_latents_w_fft, dim=(-1, -2))).real

    return init_latents_w


def eval_watermark(reversed_latents_no_w, reversed_latents_w, watermarking_mask, gt_patch, args):
    if 'complex' in args.w_measurement:
        reversed_latents_no_w_fft = torch.fft.fftshift(torch.fft.fft2(reversed_latents_no_w), dim=(-1, -2))
        reversed_latents_w_fft = torch.fft.fftshift(torch.fft.fft2(reversed_latents_w), dim=(-1, -2))
        target_patch = gt_patch
    elif 'seed' in args.w_measurement:
        reversed_latents_no_w_fft = reversed_latents_no_w
        reversed_latents_w_fft = reversed_latents_w
        target_patch = gt_patch
    else:
        NotImplementedError(f'w_measurement: {args.w_measurement}')

    if 'l1' in args.w_measurement:
        no_w_metric = torch.abs(reversed_latents_no_w_fft[watermarking_mask] - target_patch[watermarking_mask]).mean().item()
        w_metric = torch.abs(reversed_latents_w_fft[watermarking_mask] - target_patch[watermarking_mask]).mean().item()
    else:
        NotImplementedError(f'w_measurement: {args.w_measurement}')

    return no_w_metric, w_metric

def get_p_value(reversed_latents_no_w, reversed_latents_w, watermarking_mask, gt_patch, args):
    # assume it's Fourier space wm
    reversed_latents_no_w_fft = torch.fft.fftshift(torch.fft.fft2(reversed_latents_no_w), dim=(-1, -2))[watermarking_mask].flatten()
    reversed_latents_w_fft = torch.fft.fftshift(torch.fft.fft2(reversed_latents_w), dim=(-1, -2))[watermarking_mask].flatten()
    target_patch = gt_patch[watermarking_mask].flatten()

    target_patch = torch.concatenate([target_patch.real, target_patch.imag])
    
    # no_w
    reversed_latents_no_w_fft = torch.concatenate([reversed_latents_no_w_fft.real, reversed_latents_no_w_fft.imag])
    sigma_no_w = reversed_latents_no_w_fft.std()
    lambda_no_w = (target_patch ** 2 / sigma_no_w ** 2).sum().item()
    x_no_w = (((reversed_latents_no_w_fft - target_patch) / sigma_no_w) ** 2).sum().item()
    p_no_w = scipy.stats.ncx2.cdf(x=x_no_w, df=len(target_patch), nc=lambda_no_w)

    # w
    reversed_latents_w_fft = torch.concatenate([reversed_latents_w_fft.real, reversed_latents_w_fft.imag])
    sigma_w = reversed_latents_w_fft.std()
    lambda_w = (target_patch ** 2 / sigma_w ** 2).sum().item()
    x_w = (((reversed_latents_w_fft - target_patch) / sigma_w) ** 2).sum().item()
    p_w = scipy.stats.ncx2.cdf(x=x_w, df=len(target_patch), nc=lambda_w)

    return p_no_w, p_w





