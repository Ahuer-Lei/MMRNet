import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import cv2
import random
from scipy.ndimage.filters import gaussian_filter

  

def histgram_shift(data):
    num_control_point = random.randint(2,8)
    reference_control_points = torch.linspace(-1, 1, num_control_point)
    floating_control_points = reference_control_points.clone()
    for i in range(1, num_control_point - 1):
        floating_control_points[i] = floating_control_points[i - 1] + torch.rand(
            1) * (floating_control_points[i + 1] - floating_control_points[i - 1])
    img_min, img_max = data.min(), data.max()
    reference_control_points_scaled = (reference_control_points *
                                       (img_max - img_min) + img_min).numpy()
    floating_control_points_scaled = (floating_control_points *
                                      (img_max - img_min) + img_min).numpy()
    data_shifted = np.interp(data, reference_control_points_scaled,
                             floating_control_points_scaled)
    return data_shifted


def add_gaussian_noise(data, mean=0, std=0.3):
    image_shape = data.shape
    noise = torch.normal(mean, std, size=image_shape)
    vmin, vmax = torch.min(data), torch.max(data)
    mean, std = torch.mean(data), torch.std(data)
    data_normed = (data - mean) / std + noise
    data_normed = torch.clip(data_normed * std + mean, vmin, vmax)
    return data_normed





class Transformer3D(nn.Module):
    def __init__(self):
        super(Transformer3D, self).__init__()

    def forward(self, src, flow, padding_mode="border"):
        b = flow.shape[0]
        size = flow.shape[2:]
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)
        grid = grid.to(torch.float32)
        grid = grid.repeat(b, 1, 1, 1, 1)
        new_locs = grid + flow
        shape = flow.shape[2:]
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)
        new_locs = new_locs.permute(0, 2, 3, 4, 1)
        new_locs = new_locs[..., [2, 1, 0]]
        warped = F.grid_sample(src, new_locs, align_corners=True, padding_mode=padding_mode)

        return warped


class Transformer2D(nn.Module):
    def __init__(self):
        super(Transformer2D, self).__init__()

    def forward(self, imgs, tp):
        theta = tp.reshape(-1,2,3)
        theta = theta.to(torch.float32)
        size = imgs[0].unsqueeze(0).repeat(imgs.shape[0],1,1,1).size()
        grid = F.affine_grid(theta, size, align_corners=True)
        imgs_warp = F.grid_sample(imgs, grid, align_corners=True, padding_mode="zeros")
        return imgs_warp



class Transformer2D_nonrigid(nn.Module):
    def __init__(self):
        super(Transformer2D, self).__init__()

    def forward(self, src, flow, padding_mode="border"):
        b = flow.shape[0]
        size = flow.shape[2:]
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)
        grid = grid.to(torch.float32)
        grid = grid.repeat(b, 1, 1, 1).to(flow.device)
        new_locs = grid + flow
        shape = flow.shape[2:]
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)
        new_locs = new_locs.permute(0, 2, 3, 1)
        new_locs = new_locs[..., [1, 0]]
        warped = F.grid_sample(src, new_locs, align_corners=True, padding_mode=padding_mode)
        return warped


def create_affine_transformation_matrix(n_dims, scaling=None, rotation=None, shearing=None, translation=None):
    """
        create a 4x4 affine transformation matrix from specified values
    :param n_dims: integer
    :param scaling: list of 3 scaling values
    :param rotation: list of 3 angles (degrees) for rotations around 1st, 2nd, 3rd axis
    :param shearing: list of 6 shearing values
    :param translation: list of 3 values
    :return: 4x4 numpy matrix
    """

    trans_scaling = np.eye(n_dims + 1)
    trans_shearing = np.eye(n_dims + 1)
    trans_translation = np.eye(n_dims + 1)

    if scaling is not None:
        trans_scaling[np.arange(n_dims + 1), np.arange(n_dims + 1)] = np.append(scaling, 1)

    if shearing is not None:
        shearing_index = np.ones((n_dims + 1, n_dims + 1), dtype='bool')
        shearing_index[np.eye(n_dims + 1, dtype='bool')] = False
        shearing_index[-1, :] = np.zeros((n_dims + 1))
        shearing_index[:, -1] = np.zeros((n_dims + 1))
        trans_shearing[shearing_index] = shearing

    if translation is not None:
        trans_translation[np.arange(n_dims), n_dims *
                          np.ones(n_dims, dtype='int')] = translation

    if n_dims == 2:
        if rotation is None:
            rotation = np.zeros(1)
        else:
            rotation = np.asarray(rotation) * (math.pi / 180)
        trans_rot = np.eye(n_dims + 1)
        trans_rot[np.array([0, 1, 0, 1]), np.array([0, 0, 1, 1])] = [np.cos(rotation), np.sin(rotation),
                                                                     np.sin(rotation) * -1, np.cos(rotation)]
        return trans_translation @ trans_rot @ trans_shearing @ trans_scaling

    else:
        if rotation is None:
            rotation = np.zeros(n_dims)
        else:
            rotation = np.asarray(rotation) * (math.pi / 180)
        trans_rot1 = np.eye(n_dims + 1)
        trans_rot1[np.array([1, 2, 1, 2]), np.array([1, 1, 2, 2])] = [np.cos(rotation[0]),
                                                                      np.sin(
                                                                          rotation[0]),
                                                                      np.sin(
                                                                          rotation[0]) * -1,
                                                                      np.cos(rotation[0])]
        trans_rot2 = np.eye(n_dims + 1)
        trans_rot2[np.array([0, 2, 0, 2]), np.array([0, 0, 2, 2])] = [np.cos(rotation[1]),
                                                                      np.sin(
                                                                          rotation[1]) * -1,
                                                                      np.sin(
                                                                          rotation[1]),
                                                                      np.cos(rotation[1])]
        trans_rot3 = np.eye(n_dims + 1)
        trans_rot3[np.array([0, 1, 0, 1]), np.array([0, 0, 1, 1])] = [np.cos(rotation[2]),
                                                                      np.sin(
                                                                          rotation[2]),
                                                                      np.sin(
                                                                          rotation[2]) * -1,
                                                                      np.cos(rotation[2])]
        return trans_translation @ trans_rot3 @ trans_rot2 @ trans_rot1 @ trans_shearing @ trans_scaling


def non_affine_2d(imgs, padding_modes, opt, elastic_random=None):
    if not isinstance(imgs, list) and not isinstance(imgs, tuple):
        imgs = [imgs]
    if not isinstance(padding_modes, list) and not isinstance(padding_modes, tuple):
        padding_modes = [padding_modes]

    w, h = imgs[0].shape[-2:]
    if elastic_random is None:
        elastic_random = torch.rand([2, w, h]).numpy() * 2 - 1  # .numpy()

    sigma = opt['gaussian_smoothing']   # 需要根据图像大小调整
    alpha = opt['non_affine_alpha']  # 需要根据图像大小调整

    dx = gaussian_filter(elastic_random[0], sigma) * alpha
    dy = gaussian_filter(elastic_random[1], sigma) * alpha
    dx = np.expand_dims(dx, 0)
    dy = np.expand_dims(dy, 0)
    flow = np.concatenate((dx, dy), 0)
    flow = np.expand_dims(flow, 0)
    flow = torch.from_numpy(flow).to(torch.float32)

    results = []
    for img, mode in zip(imgs, padding_modes):
        img = Transformer2D_nonrigid()(img.unsqueeze(0), flow, padding_mode=mode)
        results.append(img.squeeze(0))

    return results[0] if len(results) == 1 else results


def non_affine_3d(imgs, padding_modes, opt, elastic_random=None):
    if not isinstance(imgs, list) and not isinstance(imgs, tuple):
        imgs = [imgs]
    if not isinstance(padding_modes, list) and not isinstance(padding_modes, tuple):
        padding_modes = [padding_modes]

    z, w, h = imgs[0].shape[-3:]
    if elastic_random is None:
        elastic_random = torch.rand([3, z, w, h]).numpy() * 2 - 1  # .numpy()

    sigma = opt['gaussian_smoothing']  # 需要根据图像大小调整
    alpha = opt['non_affine_alpha']  # 需要根据图像大小调整

    dz = gaussian_filter(elastic_random[0], sigma) * alpha
    dx = gaussian_filter(elastic_random[1], sigma) * alpha
    dy = gaussian_filter(elastic_random[2], sigma) * alpha

    dz = np.expand_dims(dz, 0)
    dx = np.expand_dims(dx, 0)
    dy = np.expand_dims(dy, 0)

    flow = np.concatenate((dz, dx, dy), 0)
    flow = np.expand_dims(flow, 0)
    flow = torch.from_numpy(flow).to(torch.float32)

    results = []
    for img, mode in zip(imgs, padding_modes):
        img = Transformer3D()(img.unsqueeze(0), flow, padding_mode=mode)
        results.append(img.squeeze(0))

    return results[0] if len(results) == 1 else results


def affine(random_numbers, imgs, padding_modes, opt):
    if not isinstance(imgs, list) and not isinstance(imgs, tuple):
        imgs = [imgs]
    if not isinstance(padding_modes, list) and not isinstance(padding_modes, tuple):
        padding_modes = [padding_modes]

    if opt['dim'] == 3:
        tmp = np.ones(3)
        tmp[0:3] = random_numbers[0:3]
        scaling = tmp * opt['scaling'] + 1
        tmp[0:3] = random_numbers[3:6]
        rotation = tmp * opt['rotation']
        tmp[0:2] = random_numbers[6:8]
        tmp[2] = 0
        translation = tmp * opt['translation']
    else:
        scaling = random_numbers[0:2] * opt['scaling'] + 1
        rotation = random_numbers[2] * opt['rotation']
        translation = random_numbers[3] * opt['translation']

    theta = create_affine_transformation_matrix(
        n_dims=opt['dim'], scaling=scaling, rotation=rotation, shearing=None, translation=translation)
    
    # 计算扭曲后的四角点
    four_corners = np.array([[0, 0], [0, 255], [255, 0], [255, 255]], dtype=np.float32).reshape(-1, 1, 2)
    T = np.array([[2 / 256, 0, -1],
                  [0, 2 / 256, -1],
                  [0, 0, 1]])
    
    matrix_warp = np.linalg.inv(T) @ np.linalg.inv(theta) @ T
    new_four_point = cv2.transform(four_corners, matrix_warp[:-1,:])
    new_four_point = torch.from_numpy(new_four_point).to(torch.float32)

    # 计算配准gt_tp
    matrix = np.linalg.inv(theta)
    gt_tp = matrix[:-1, :]
    

    theta = theta[:-1, :]
    theta = torch.from_numpy(theta).to(torch.float32)
    size = imgs[0].size()
    grid = F.affine_grid(theta.unsqueeze(0), size, align_corners=True)

    res_img = []
    for img, mode in zip(imgs, padding_modes):
        res_img.append(F.grid_sample(img, grid, align_corners=True, padding_mode=mode).squeeze(0))

    return res_img[0] if len(res_img) == 1 else res_img, gt_tp, new_four_point
