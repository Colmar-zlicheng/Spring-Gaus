# tune multi-threading params
import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import cv2

cv2.setNumThreads(0)

import torch
import json
import random
import argparse
import imageio
import numpy as np
import lib.models
from time import time
from random import randint
from termcolor import colored
from lib.utils.etqdm import etqdm
from lib.utils.logger import logger
from lib.utils.config import get_config_merge_default
from lib.utils.misc import bar_perfixes, format_args_cfg
from lib.utils.net_utils import setup_seed
from lib.utils.recorder import Recorder
from lib.models.gaus import Scene, GaussianModel_isotropic, render
from train import config_parser, get_simulator


@torch.no_grad()
def eval_all(args, cfg, scene: Scene, recorder: Recorder):
    if cfg.REAL:
        with open(os.path.join(cfg.CHECKPOINTS_ROOT, f'init_velocity_{cfg.VELOCITY.ITERATIONS}_seq{cfg.DATA.SEQ}.json'),
                  'r') as file:
            load_velocity = json.load(file)
        with open(os.path.join(cfg.CHECKPOINTS_ROOT, f'g_{cfg.VELOCITY.ITERATIONS}_seq{cfg.DATA.SEQ}.json'),
                  'r') as file:
            load_g = json.load(file)
    else:
        with open(os.path.join(cfg.CHECKPOINTS_ROOT, f'init_velocity_{cfg.VELOCITY.ITERATIONS}.json'), 'r') as file:
            load_velocity = json.load(file)
        load_g = None

    simulator, gaussians = get_simulator(args,
                                         cfg,
                                         scene,
                                         cfg_stage=cfg.DYNAMIC,
                                         init_velocity=load_velocity,
                                         load_g=load_g)
    simulator.eval()
    background = torch.tensor(scene.dataset.bg, dtype=torch.float32, device="cuda")

    xyz_all = simulator.init_xyz_all.detach().clone()
    xyz = simulator.init_xyz.detach().clone()
    v = simulator.init_v.detach().clone()
    gaussians._xyz = xyz_all

    xyz_all = torch.sum(xyz[simulator.intrp_index] * simulator.intrp_coef.unsqueeze(-1), dim=1)
    gaussians._xyz = xyz_all

    for frame_id in etqdm(range(scene.dataset.all_frames)):
        for cam_id in range(scene.dataset.num_views):
            viewpoint_cam = scene.getEvalCameras(0, cam_id)

            render_pkg = render(
                viewpoint_cam,
                gaussians,
                background,
                override_color=gaussians.get_color,
                debug=False,
                compute_cov3D_python=args.compute_cov3D_python,
                convert_SHs_python=args.convert_SHs_python,
            )
            render_image = render_pkg["render"]
            img_wirte_dir = os.path.join(scene.exp_path, f'evaluations/render/cam_{cam_id}')
            os.makedirs(img_wirte_dir, exist_ok=True)
            imageio.imwrite(os.path.join(img_wirte_dir, f"{frame_id:03}.png"),
                            (render_image.detach().cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8))

        pcd = o3d.geometry.PointCloud()
        if args.viz_anchor:
            pcd.points = o3d.utility.Vector3dVector(xyz.detach().cpu().numpy())
        else:
            pcd.points = o3d.utility.Vector3dVector(xyz_all.detach().cpu().numpy())
        ply_write_dir = os.path.join(scene.exp_path, 'evaluations/simulate')
        os.makedirs(ply_write_dir, exist_ok=True)
        o3d.io.write_point_cloud(os.path.join(ply_write_dir, f"{frame_id:03}.ply"), pcd)

        xyz_all, xyz, v, _ = simulator(xyz_all, xyz, v, frame_id + 1)
        gaussians._xyz = xyz_all


if __name__ == '__main__':
    exp_time = time()
    arg = config_parser()
    assert arg.dy_reload is not None
    cfg = get_config_merge_default(config_file=arg.cfg, arg=arg)

    logger.warning(f"final args and cfg: \n{format_args_cfg(arg, cfg)}")

    os.environ['CUDA_VISIBLE_DEVICES'] = arg.gpu_id
    import open3d as o3d  # Must import open3d after set CUDA_VISIBLE_DEVICES!

    setup_seed(cfg.SEED)
    recorder = Recorder(arg.exp_id, cfg, rank=0, time_f=exp_time)
    exp_path = f"{recorder.exp_id}_{recorder.timestamp}"

    scene = Scene(cfg, exp_path, shuffle=False, load_static=False)

    eval_all(arg, cfg, scene, recorder)
