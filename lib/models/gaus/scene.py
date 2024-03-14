import os
import random
import json
import numpy as np
import trimesh
import torch
from typing import NamedTuple
from plyfile import PlyData, PlyElement
from lib.datasets import load_dataset
from lib.utils.logger import logger
from lib.utils.etqdm import etqdm
from . import GaussianModel_isotropic
from .utils.camera_utils import cameraList_from_camInfos, camera_to_JSON
from .utils.graphics_utils import BasicPointCloud, getWorld2View2
from .utils.sh_utils import SH2RGB


def load_vertex_np(filename):
    mesh = trimesh.load(filename)
    v = mesh.vertices
    return np.array(v, dtype=np.float32)


def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)


def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'), ('red', 'u1'),
             ('green', 'u1'), ('blue', 'u1')]

    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)


def getNerfppNorm(cam_info):

    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}


class Scene():
    gaussians: GaussianModel_isotropic

    def __init__(self, cfg, exp_path, shuffle=True, load_static=True) -> None:
        self.name = type(self).__name__
        self.cfg = cfg
        self.exp_path = os.path.join('./exp', exp_path)
        self.shuffle = shuffle

        self.dataset = load_dataset(cfg.DATA)
        self._getSceneInfo(load_static=load_static)

    def _getSceneInfo(self, load_static=True, real_must_img=False):
        if self.cfg.REAL and load_static:
            self.get_static_sence()

        cameras_extent_all = []
        self.train_cameras_all = []
        self.cameras_all = []
        camlist = []

        if real_must_img:
            cam_infos_all = self.dataset.get_cam_info(img_is_mask=False)
        else:
            cam_infos_all = self.dataset.get_cam_info()

        logger.info("Loading Scene Info...")
        for frame_id in etqdm(range(self.dataset.n_frames)):
            train_cam_infos = cam_infos_all[frame_id]
            nerf_normalization = getNerfppNorm(train_cam_infos)

            camlist.extend(train_cam_infos)

            self.cameras_all.append(cameraList_from_camInfos(train_cam_infos, 1.0))

            if self.shuffle:
                random.shuffle(train_cam_infos)

            cameras_extent_all.append(nerf_normalization["radius"])
            self.train_cameras_all.append(cameraList_from_camInfos(train_cam_infos, 1.0))

        json_cams = []
        for id, cam in enumerate(camlist):
            json_cams.append(camera_to_JSON(id, cam))
        with open(os.path.join(self.exp_path, "cameras.json"), 'w') as file:
            json.dump(json_cams, file, indent=4)

        if not self.cfg.REAL:
            self.static_cameras_extent_all = cameras_extent_all[0]

        self.dynamic_cameras_extent_all = cameras_extent_all[0]

    def get_static_sence(self):
        static_cam_infos = self.dataset.get_colmap_info()

        self.static_cam_all = cameraList_from_camInfos(static_cam_infos, 1.0)

        static_nerf_normalization = getNerfppNorm(static_cam_infos)
        self.static_cameras_extent_all = static_nerf_normalization["radius"]

        json_cams = []
        for id, cam in enumerate(static_cam_infos):
            json_cams.append(camera_to_JSON(id, cam))
        with open(os.path.join(self.exp_path, "cameras_static.json"), 'w') as file:
            json.dump(json_cams, file, indent=4)

    def set_gaussians(self, gaussians: GaussianModel_isotropic, load_path=None):
        self.gaussians = gaussians

        if load_path is None:
            if self.cfg.REAL:
                colmap_pcd_path = os.path.join(self.dataset.static_dir,
                                               f'colmap/{self.dataset.obj_name}/colmap_pcd.ply')
                colmap_pcd = load_vertex_np(colmap_pcd_path)
                pcd = self.random_points(xyz_max=np.max(colmap_pcd, axis=0), xyz_min=np.min(colmap_pcd, axis=0))
            else:
                pcd = self.random_points(xyz_max=np.array(self.cfg.DATA.XYZ_MAX),
                                         xyz_min=np.array(self.cfg.DATA.XYZ_MIN))
            self.gaussians.create_from_pcd(pcd, self.static_cameras_extent_all)
        else:
            logger.info(f"Load gaussians from {load_path}...")
            self.gaussians.load_ply(load_path)

            if self.cfg.REAL:
                self.gaussians.max_radii2D = torch.zeros((self.gaussians.get_xyz.shape[0]), device="cuda")
                self.gaussians.spatial_lr_scale = self.dynamic_cameras_extent_all

    def save(self, iteration, stage='static', save_path=None):
        if save_path is None:
            point_cloud_path = os.path.join(self.exp_path, "point_cloud/{}/iteration_{}".format(stage, iteration))
        else:
            point_cloud_path = save_path
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    def random_points(self, xyz_max, xyz_min, num_pts=100000):
        logger.info(f"Generating random point cloud ({num_pts})...")

        xyz = np.random.random((num_pts, 3)) * (xyz_max - xyz_min) + xyz_min
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        ply_path = os.path.join(self.exp_path, "input_points.ply")
        storePly(ply_path, xyz, SH2RGB(shs) * 255)

        return fetchPly(ply_path)

    def getTrainCameras(self, frame_id):
        return self.train_cameras_all[frame_id]

    def getEvalCameras(self, frame_id, cam_id):
        # logger.warning("Please check all the cameras are fixed!")
        return self.cameras_all[frame_id][cam_id]

    def getStaticCameras(self):
        if self.cfg.REAL:
            return self.static_cam_all
        else:
            return self.getTrainCameras(0)
