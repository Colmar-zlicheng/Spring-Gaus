import math
import random
from copy import deepcopy
from re import L
from typing import List, Optional, Union

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as tvF
from pytorch3d.transforms import (
    axis_angle_to_matrix,
    axis_angle_to_quaternion,
    euler_angles_to_matrix,
    matrix_to_euler_angles,
    matrix_to_quaternion,
    matrix_to_rotation_6d,
    quaternion_to_axis_angle,
    quaternion_to_matrix,
    rotation_6d_to_matrix,
)

from .logger import logger


class Compose:

    def __init__(self, transforms: list):
        """Composes several transforms together. This transform does not
        support torchscript.

        Args:
            transforms (list): (list of transform functions)
        """
        self.transforms = transforms

    def __call__(self, rotation: Union[torch.Tensor, np.ndarray], convention: str = "xyz", **kwargs):
        convention = convention.lower()
        if not (set(convention) == set("xyz") and len(convention) == 3):
            raise ValueError(f"Invalid convention {convention}.")
        if isinstance(rotation, np.ndarray):
            data_type = "numpy"
            rotation = torch.FloatTensor(rotation)
        elif isinstance(rotation, torch.Tensor):
            data_type = "tensor"
        else:
            raise TypeError("Type of rotation should be torch.Tensor or numpy.ndarray")
        for t in self.transforms:
            if "convention" in t.__code__.co_varnames:
                rotation = t(rotation, convention.upper(), **kwargs)
            else:
                rotation = t(rotation, **kwargs)
        if data_type == "numpy":
            rotation = rotation.detach().cpu().numpy()
        return rotation


def aa_to_rotmat(axis_angle: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
    """
    Convert axis_angle to rotation matrixs.
    Args:
        axis_angle (Union[torch.Tensor, numpy.ndarray]): input shape
                should be (..., 3). ndim of input is unlimited.

    Returns:
        Union[torch.Tensor, numpy.ndarray]: shape would be (..., 3, 3).
    """
    if axis_angle.shape[-1] != 3:
        raise ValueError(f"Invalid input axis angles shape f{axis_angle.shape}.")
    t = Compose([axis_angle_to_matrix])
    return t(axis_angle)


def rotmat_to_aa(matrix: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
    """Convert rotation matrixs to axis angles.

    Args:
        matrix (Union[torch.Tensor, numpy.ndarray]): input shape
                should be (..., 3, 3). ndim of input is unlimited.
        convention (str, optional): Convention string of three letters
                from {“x”, “y”, and “z”}. Defaults to 'xyz'.

    Returns:
        Union[torch.Tensor, numpy.ndarray]: shape would be (..., 3).
    """
    if matrix.shape[-1] != 3 or matrix.shape[-2] != 3:
        raise ValueError(f"Invalid rotation matrix  shape f{matrix.shape}.")
    t = Compose([matrix_to_quaternion, quaternion_to_axis_angle])
    return t(matrix)


def aa_to_quat(axis_angle: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
    """
    Convert axis_angle to quaternions.
    Args:
        axis_angle (Union[torch.Tensor, numpy.ndarray]): input shape
                should be (..., 3). ndim of input is unlimited.

    Returns:
        Union[torch.Tensor, numpy.ndarray]: shape would be (..., 4).
    """
    if axis_angle.shape[-1] != 3:
        raise ValueError(f"Invalid input axis angles f{axis_angle.shape}.")
    t = Compose([axis_angle_to_quaternion])
    return t(axis_angle)


def aa_to_rot6d(axis_angle: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
    """Convert axis angles to rotation 6d representations.

    Args:
        axis_angle (Union[torch.Tensor, numpy.ndarray]): input shape
                should be (..., 3). ndim of input is unlimited.

    Returns:
        Union[torch.Tensor, numpy.ndarray]: shape would be (..., 6).

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """
    if axis_angle.shape[-1] != 3:
        raise ValueError(f"Invalid input axis_angle f{axis_angle.shape}.")
    t = Compose([axis_angle_to_matrix, matrix_to_rotation_6d])
    return t(axis_angle)


def rot6d_to_aa(rotation_6d: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
    """Convert rotation 6d representations to axis angles.

    Args:
        rotation_6d (Union[torch.Tensor, numpy.ndarray]): input shape
                should be (..., 6). ndim of input is unlimited.

    Returns:
        Union[torch.Tensor, numpy.ndarray]: shape would be (..., 3).

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """
    if rotation_6d.shape[-1] != 6:
        raise ValueError(f"Invalid input rotation_6d f{rotation_6d.shape}.")
    t = Compose([rotation_6d_to_matrix, matrix_to_quaternion, quaternion_to_axis_angle])
    return t(rotation_6d)


def quat_to_aa(quaternions: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
    """Convert quaternions to axis angles.

    Args:
        quaternions (Union[torch.Tensor, numpy.ndarray]): input shape
                should be (..., 3). ndim of input is unlimited.
    Returns:
        Union[torch.Tensor, numpy.ndarray]: shape would be (..., 3).
    """
    if quaternions.shape[-1] != 4:
        raise ValueError(f"Invalid input quaternions f{quaternions.shape}.")
    t = Compose([quaternion_to_axis_angle])
    return t(quaternions)


def rot6d_to_rotmat(rotation_6d: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
    """Convert rotation 6d representations to rotation matrixs.

    Args:
        rotation_6d (Union[torch.Tensor, numpy.ndarray]): input shape
                should be (..., 6). ndim of input is unlimited.
    Returns:
        Union[torch.Tensor, numpy.ndarray]: shape would be (..., 3, 3).

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """
    if rotation_6d.shape[-1] != 6:
        raise ValueError(f"Invalid input rotation_6d f{rotation_6d.shape}.")
    t = Compose([rotation_6d_to_matrix])
    return t(rotation_6d)


def rotmat_to_rot6d(matrix: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
    """Convert rotation matrixs to rotation 6d representations.

    Args:
        matrix (Union[torch.Tensor, numpy.ndarray]): input shape
                should be (..., 3, 3). ndim of input is unlimited.
    Returns:
        Union[torch.Tensor, numpy.ndarray]: shape would be (..., 6).

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """
    if matrix.shape[-1] != 3 or matrix.shape[-2] != 3:
        raise ValueError(f"Invalid rotation matrix  shape f{matrix.shape}.")
    t = Compose([matrix_to_rotation_6d])
    return t(matrix)


def rotmat_to_quat(matrix: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
    """Convert rotation matrixs to quaternions.

    Args:
        matrix (Union[torch.Tensor, numpy.ndarray]): input shape
                should be (..., 3, 3). ndim of input is unlimited.
    Returns:
        Union[torch.Tensor, numpy.ndarray]: shape would be (..., 4).
    """
    if matrix.shape[-1] != 3 or matrix.shape[-2] != 3:
        raise ValueError(f"Invalid rotation matrix  shape f{matrix.shape}.")
    t = Compose([matrix_to_quaternion])
    return t(matrix)


def quat_to_rotmat(quaternions: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
    """Convert quaternions to rotation matrixs.

    Args:
        quaternions (Union[torch.Tensor, numpy.ndarray]): input shape
                should be (..., 3). ndim of input is unlimited.
    Returns:
        Union[torch.Tensor, numpy.ndarray]: shape would be (..., 3, 3).
    """
    if quaternions.shape[-1] != 4:
        raise ValueError(f"Invalid input quaternions shape f{quaternions.shape}.")
    t = Compose([quaternion_to_matrix])
    return t(quaternions)


def quat_to_rot6d(quaternions: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
    """Convert quaternions to rotation 6d representations.

    Args:
        quaternions (Union[torch.Tensor, numpy.ndarray]): input shape
                should be (..., 4). ndim of input is unlimited.

    Returns:
        Union[torch.Tensor, numpy.ndarray]: shape would be (..., 6).

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """
    if quaternions.shape[-1] != 4:
        raise ValueError(f"Invalid input quaternions f{quaternions.shape}.")
    t = Compose([quaternion_to_matrix, matrix_to_rotation_6d])
    return t(quaternions)


def rot6d_to_quat(rotation_6d: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
    """Convert rotation 6d representations to quaternions.

    Args:
        rotation (Union[torch.Tensor, numpy.ndarray]): input shape
                should be (..., 6). ndim of input is unlimited.

    Returns:
        Union[torch.Tensor, numpy.ndarray]: shape would be (..., 4).

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """
    if rotation_6d.shape[-1] != 6:
        raise ValueError(f"Invalid input rotation_6d shape f{rotation_6d.shape}.")
    t = Compose([rotation_6d_to_matrix, matrix_to_quaternion])
    return t(rotation_6d)


def euler_to_quat(euler: Union[torch.Tensor, np.ndarray], convention: str = "XYZ") -> Union[torch.Tensor, np.ndarray]:
    """Convert Euler angles to quaternions.

    Args:
        euler (Union[torch.Tensor, numpy.ndarray]): input shape
                should be (..., 3). ndim of input is unlimited.
        convention (str, optional): Convention string of three letters
                from {“X”, “Y”, and “Z”}. Defaults to 'XYZ'.

    Returns:
        Union[torch.Tensor, numpy.ndarray]: shape would be (..., 4).
    """
    if euler.shape[-1] != 3:
        raise ValueError(f"Invalid input euler shape f{euler.shape}.")
    t = Compose([euler_angles_to_matrix, matrix_to_quaternion])
    return t(euler, convention)


def quat_to_euler(quaternions: Union[torch.Tensor, np.ndarray],
                  convention: str = "XYZ") -> Union[torch.Tensor, np.ndarray]:
    """Convert quaternions to Euler angles.

    Args:
        quaternions (Union[torch.Tensor, numpy.ndarray]): input shape
                should be (..., 4). ndim of input is unlimited.
        convention (str, optional): Convention string of three letters
                from {“X”, “Y”, and “Z”}. Defaults to 'XYZ'.

    Returns:
        Union[torch.Tensor, numpy.ndarray]: shape would be (..., 3).
    """
    if quaternions.shape[-1] != 4:
        raise ValueError(f"Invalid input quaternions shape f{quaternions.shape}.")
    t = Compose([quaternion_to_matrix, matrix_to_euler_angles])
    return t(quaternions, convention)


def batch_cam_extr_transf(batch_cam_extr, batch_joints):
    """apply batch camera extrinsic transformation on batch joints

    Args:
        batch_cam_extr (torch.Tensor): shape (BATCH, NPERSP, 4, 4)
        batch_joints (torch.Tensor): shape (BATCH, NPERSP, NJOINTS, 3)

    Returns:
        torch.Tensor: shape (BATCH, NPERSP, NJOINTS, 3)
    """
    res = (batch_cam_extr[..., :3, :3] @ batch_joints.transpose(-2, -1)).transpose(-2, -1)
    # [B, NPERSP, 3, 3] @ [B, NPERSP, 3, 21] => [B, NPERSP, 3, 21] => [B, NPERSP, 21, 3]
    res = res + batch_cam_extr[..., :3, 3].unsqueeze(-2)
    return res


def batch_cam_intr_projection(batch_cam_intr, batch_joints, eps=1e-7):
    """apply camera projection on batch joints with batch intrinsics

    Args:
        batch_cam_intr (torch.Tensor): shape (BATCH, NPERSP, 3, 3)
        batch_joints (torch.Tensor): shape (BATCH, NPERSP, NJOINTS, 3)
        eps (float, optional): avoid divided by zero. Defaults to 1e-7.

    Returns:
        torch.Tensor: shape (BATCH, NPERSP, NJOINTS, 2)
    """
    res = (batch_cam_intr @ batch_joints.transpose(-2, -1)).transpose(-2, -1)  # [B, NPERSP, 21, 3]
    xy = res[..., 0:2]
    z = res[..., 2:]
    z[torch.abs(z) < eps] = eps
    uv = xy / z
    return uv


def batch_persp_project(verts: torch.Tensor, camintr: torch.Tensor):
    """Batch apply perspective procjection on points

    Args:
        verts (torch.Tensor): 3D points with shape (B, N, 3)
        camintr (torch.Tensor): intrinsic matrix with shape (B, 3, 3)

    Returns:
        torch.Tensor: shape (B, N, 2)
    """
    # Project 3d vertices on image plane
    verts_hom2d = camintr.bmm(verts.transpose(1, 2)).transpose(1, 2)
    proj_verts2d = verts_hom2d[:, :, :2] / verts_hom2d[:, :, 2:]
    return proj_verts2d


def persp_project(points3d, cam_intr):
    """Apply perspective camera projection on a 3D point

    Args:
        points3d (np.ndarray): shape (N, 3)
        cam_intr (np.ndarray): shape (3, 3)

    Returns:
        np.ndarray: shape (N, 2)
    """
    hom_2d = np.array(cam_intr).dot(points3d.transpose()).transpose()
    points2d = (hom_2d / (hom_2d[:, 2:] + 1e-6))[:, :2]
    return points2d.astype(np.float32)


def SE3_transform(points3d, transform):
    """Apply SE3 transform on a 3D point

    Args:
        points3d (np.ndarray): shape (N, 3)
        transform (np.ndarray): shape (4, 4)

    Returns:
        np.ndarray: shape (N, 3)
    """
    return (transform[:3, :3] @ points3d.T).T + transform[:3, 3][None, :]


def persp_project_torch(points3d, cam_intr):
    """Apply perspective camera projection on a 3D point

    Args:
        points3d (torch.Tensor): shape (N, 3)
        cam_intr (torch.Tensor): shape (3, 3)

    Returns:
        torch.Tensor: shape (N, 2)
    """
    hom_2d = cam_intr.mm(points3d.transpose(0, 1)).transpose(0, 1)
    points2d = (hom_2d / (hom_2d[:, 2:] + 1e-6))[:, :2]
    return points2d


def SE3_transform_torch(points3d, transform):
    """Apply SE3 transform on a 3D point

    Args:
        points3d (torch.Tensor): shape (N, 3)
        transform (torch.Tensor): shape (4, 4)

    Returns:
        torch.Tensor: shape (N, 3)
    """
    return (transform[:3, :3] @ points3d.transpose(0, 1)).transpose(0, 1) + transform[:3, 3][None, :]


def inverse_sigmoid(x, eps=1e-5):
    """Inverse function of sigmoid.

    Args:
        x (Tensor): The tensor to do the
            inverse.
        eps (float): EPS avoid numerical
            overflow. Defaults 1e-5.
    Returns:
        Tensor: The x has passed the inverse
            function of sigmoid, has same
            shape with input.
    """
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)


def uniform_sampling(cloud_points, voxel_size):
    scaled_points = cloud_points / voxel_size
    scaled_points = torch.floor(scaled_points).int() + 0.5

    uniform_grid = torch.unique(scaled_points, dim=0)
    uniform_cloud = uniform_grid * voxel_size

    return uniform_cloud
