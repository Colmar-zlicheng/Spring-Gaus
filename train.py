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
from lib.utils.net_utils import setup_seed, clip_gradient
from lib.utils.recorder import Recorder
from lib.utils.transform import inverse_sigmoid, uniform_sampling
from lib.models.gaus import Scene, GaussianModel_isotropic, render
from lib.models.regist import Register
from lib.utils.builder import build_simulator
from lib.metrics.img_loss import l1_loss, ssim
from lib.utils.video_utils import image_to_video


def config_parser():
    import ast

    def parse_nested_list(input_string):
        try:
            return ast.literal_eval(input_string)
        except:
            raise argparse.ArgumentTypeError("Cannot parse as a list")

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("-c", "--cfg", help="experiment configure file name", type=str, default=None)
    parser.add_argument("-exp", "--exp_id", default="default", type=str, help="Experiment ID")
    parser.add_argument("-g", "--gpu_id", type=str, default='0', help="override enviroment var CUDA_VISIBLE_DEVICES")
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--dy_reload', type=str, default=None)
    parser.add_argument('--v_reload', type=str, default=None)
    parser.add_argument('--eval_cam', type=int, default=-1)
    parser.add_argument('--global_k', type=float)
    parser.add_argument('--compute_cov3D_python', action='store_true')
    parser.add_argument('--convert_SHs_python', action='store_true')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--viz_anchor', action='store_true')
    parser.add_argument('--boundary_condition', default=None, type=parse_nested_list)

    return parser.parse_args()


def train_static(args, cfg, scene: Scene, save_root):
    cfg_static = cfg.STATIC
    viewpoint_stack = None
    ema_loss_for_log = 0.0

    gaussians = GaussianModel_isotropic(const_scale=cfg_static.CONST_SCALE)

    scene.set_gaussians(gaussians, None)
    gaussians.training_setup(cfg_static)

    background = torch.tensor(scene.dataset.bg, dtype=torch.float32, device="cuda")

    train_bar = etqdm(range(1, cfg_static.ITERATIONS + 1))
    for iteration in train_bar:
        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getStaticCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

        # Render
        render_pkg = render(
            viewpoint_cam,
            gaussians,
            background,
            override_color=gaussians.get_color,
            debug=True if (iteration - 1) >= args.debug_from else False,
            compute_cov3D_python=args.compute_cov3D_python,
            convert_SHs_python=args.convert_SHs_python,
        )
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg[
            "viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - cfg_static.LAMBDA_DSSIM) * Ll1 + cfg_static.LAMBDA_DSSIM * (1.0 - ssim(image, gt_image))
        loss.backward()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                train_bar.set_description(f"{bar_perfixes['static']} loss: {ema_loss_for_log:.{7}f}")

            # visualization
            if iteration % cfg_static.VIZ_INTERVAL == 0:
                img_list = [
                    image.detach().cpu().permute(1, 2, 0).numpy() * 255,
                    gt_image.detach().cpu().permute(1, 2, 0).numpy() * 255
                ]
                img_list = np.hstack(img_list).astype(np.uint8)
                img_write_dir = os.path.join(scene.exp_path, 'viz_static')
                os.makedirs(img_write_dir, exist_ok=True)
                imageio.imwrite(os.path.join(img_write_dir, f"{iteration}.png"), img_list)

            # Log and save
            if (iteration in cfg_static.SAVE_ITERATIONS):
                print("")
                logger.info("[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if iteration < cfg_static.DENSIFY_UNTIL_ITER:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter],
                                                                     radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > cfg_static.DENSIFY_FROM_ITER and iteration % cfg_static.DENSIFICATION_INTERVAL == 0:
                    size_threshold = 20 if iteration > cfg_static.OPACITY_RESET_INTERVAL else None
                    gaussians.densify_and_prune(cfg_static.DENSIFY_GRAD_THRESHOLD, cfg_static.GRAD_SHRESHOLD,
                                                scene.static_cameras_extent_all, size_threshold)

                if iteration % cfg_static.OPACITY_RESET_INTERVAL == 0 or (scene.dataset.white_bkg and
                                                                          iteration == cfg_static.DENSIFY_FROM_ITER):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < cfg_static.ITERATIONS:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)

            if (iteration in cfg_static.CHECKPOINT_ITERATIONS):
                print("")
                logger.info(f"[ITER {iteration}] Saving Checkpoint")
                os.makedirs(os.path.join(scene.exp_path, 'checkpoints_static'), exist_ok=True)
                torch.save((gaussians.capture(), iteration),
                           os.path.join(scene.exp_path, 'checkpoints_static', f"chkpnt{iteration}.pth"))

        torch.cuda.empty_cache()

    with torch.no_grad():
        print("")
        logger.warning("Saving Static Gaussians")
        os.makedirs(save_root, exist_ok=True)

        static_name = 'static_gaussians'

        save_path = os.path.join(save_root, static_name)
        scene.save(iteration, save_path=save_path)


def register_gaus(args, cfg, scene: Scene, save_root):
    from simple_knn._C import distCUDA2

    cfg_regist = cfg.REGIST
    viewpoint_stack = None
    ema_loss_for_log = 0.0

    gaussians = GaussianModel_isotropic()

    scene.set_gaussians(gaussians, os.path.join(save_root, 'static_gaussians/point_cloud.ply'))
    background = torch.tensor(scene.dataset.bg, dtype=torch.float32, device="cuda")

    xyz_i = gaussians.get_xyz.detach().clone()
    dist2 = torch.clamp_min(distCUDA2(xyz_i * cfg.DATA.REGIST.INIT_S[0]), 0.0000001)
    scales = torch.log(torch.sqrt(dist2[..., None]))
    gaussians._scaling = torch.nn.Parameter(scales, requires_grad=True)

    regist = Register(cfg_regist, cfg.DATA.REGIST).cuda()
    regist.training_setup(cfg_regist)

    train_bar = etqdm(range(1, cfg_regist.ITERATIONS + 1))
    for iteration in train_bar:
        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras(0).copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

        gaussians._xyz = regist(xyz_i)

        dist2 = torch.clamp_min(distCUDA2(xyz_i * regist.s), 0.0000001)
        scales = torch.log(torch.sqrt(dist2[..., None]))
        gaussians._scaling = scales

        # Render
        render_pkg = render(
            viewpoint_cam,
            gaussians,
            background,
            override_color=gaussians.get_color,
            debug=False,
            compute_cov3D_python=args.compute_cov3D_python,
            convert_SHs_python=args.convert_SHs_python,
        )
        image = render_pkg["render"]

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - cfg_regist.LAMBDA_DSSIM) * Ll1 + cfg_regist.LAMBDA_DSSIM * (1.0 - ssim(image, gt_image))
        loss.backward()

        regist.optimizer.step()
        regist.optimizer.zero_grad(set_to_none=True)

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                train_bar.set_description(f"{bar_perfixes['regist']} loss: {ema_loss_for_log:.{7}f}")

            # visualization
            if iteration % cfg_regist.VIZ_INTERVAL == 0:
                img_list = [
                    image.detach().cpu().permute(1, 2, 0).numpy() * 255,
                    gt_image.detach().cpu().permute(1, 2, 0).numpy() * 255
                ]
                img_list = np.hstack(img_list).astype(np.uint8)
                img_write_dir = os.path.join(scene.exp_path, 'viz_regist')
                os.makedirs(img_write_dir, exist_ok=True)
                imageio.imwrite(os.path.join(img_write_dir, f"{iteration}.png"), img_list)

    with torch.no_grad():
        print("")
        logger.warning("Saving Registed Gaussians")
        os.makedirs(save_root, exist_ok=True)

        regist_name = f'regist_gaussians_{cfg.DATA.SEQ}'

        save_path = os.path.join(save_root, regist_name)
        scene.save(iteration, save_path=save_path)


def render_step(args,
                cfg_stage,
                iteration,
                viewpoint_cam,
                frame_id,
                gaussians: GaussianModel_isotropic,
                background,
                optim=True,
                stage='dynamic',
                **kwargs):
    # Render
    render_pkg = render(
        viewpoint_cam,
        gaussians,
        background,
        # override_color=gaussians.get_color,
        override_color=torch.ones_like(gaussians.get_color, dtype=torch.float32).cuda() if
        (stage == 'dynamic' and cfg.DATA.get("IMG_IS_MASK", False)) else gaussians.get_color,
        debug=True if (iteration - 1) >= args.debug_from else False,
        compute_cov3D_python=args.compute_cov3D_python,
        convert_SHs_python=args.convert_SHs_python,
    )
    image = render_pkg["render"]

    # Loss
    gt_image = viewpoint_cam.original_image.cuda()
    Ll1 = l1_loss(image, gt_image)
    loss = (1.0 - cfg_stage.LAMBDA_DSSIM) * Ll1 + cfg_stage.LAMBDA_DSSIM * (1.0 - ssim(image, gt_image))
    if stage != 'eval':
        if cfg_stage.LAMBDA_CENTER != 0:
            assert cfg.DATA.IMG_IS_MASK
            y_coords, x_coords = torch.meshgrid(torch.arange(0, image.shape[1], dtype=torch.float32),
                                                torch.arange(0, image.shape[2], dtype=torch.float32),
                                                indexing='ij')
            y_coords = y_coords.cuda()
            x_coords = x_coords.cuda()

            mask_pred = torch.mean(image, dim=0)
            weighted_x = (x_coords * mask_pred).sum() / mask_pred.sum()
            weighted_y = (y_coords * mask_pred).sum() / mask_pred.sum()

            mask_gt = torch.mean(gt_image, dim=0)
            weighted_x_gt = (x_coords * mask_gt).sum() / mask_gt.sum()
            weighted_y_gt = (y_coords * mask_gt).sum() / mask_gt.sum()

            center_loss = torch.sqrt((weighted_x - weighted_x_gt)**2 + (weighted_y - weighted_y_gt)**2) / image.shape[2]

            loss += cfg_stage.LAMBDA_CENTER * center_loss

        if cfg_stage.LAMBDA_PERCEP != 0:

            target_act = crit_vgg.get_features(gt_image)

            loss += cfg_stage.LAMBDA_PERCEP * crit_vgg(image, target_act, target_is_features=True)
            loss += cfg_stage.LAMBDA_PERCEP * crit_tv(image) * 20

        if optim:
            loss.backward()

    with torch.no_grad():
        # visualization
        if iteration % cfg_stage.VIZ_INTERVAL == 0:
            img_list = [
                image.detach().cpu().permute(1, 2, 0).numpy() * 255,
                gt_image.detach().cpu().permute(1, 2, 0).numpy() * 255
            ]
            img_list = np.hstack(img_list).astype(np.uint8)
            img_write_dir = os.path.join(scene.exp_path, f'viz_{stage}')
            os.makedirs(img_write_dir, exist_ok=True)
            imageio.imwrite(os.path.join(img_write_dir, f"{iteration}_{viewpoint_cam.colmap_id}_{frame_id}.png"),
                            img_list)
        if iteration == -1:
            img_list = [
                image.detach().cpu().permute(1, 2, 0).numpy() * 255,
                gt_image.detach().cpu().permute(1, 2, 0).numpy() * 255
            ]
            img_list = np.hstack(img_list).astype(np.uint8)
            img_write_dir = os.path.join(scene.exp_path, 'evaluations/images_fit')
            os.makedirs(img_write_dir, exist_ok=True)
            imageio.imwrite(os.path.join(img_write_dir, f"{viewpoint_cam.colmap_id}_{frame_id:02}.png"), img_list)
        if iteration == -2:
            img_list = image.detach().cpu().permute(1, 2, 0).numpy() * 255
            img_write_dir = os.path.join(scene.exp_path, 'evaluations/images_pred')
            os.makedirs(img_write_dir, exist_ok=True)
            imageio.imwrite(os.path.join(img_write_dir, f"{viewpoint_cam.colmap_id}_{frame_id:02}.png"),
                            img_list.astype(np.uint8))

    return loss.detach().cpu().item(), image.detach().cpu()


def get_simulator(args, cfg, scene: Scene, cfg_stage, init_velocity, load_g):
    gaussians = GaussianModel_isotropic()

    static_name = f'regist_gaussians_{cfg.DATA.SEQ}' if cfg.REAL else 'static_gaussians'

    if args.dy_reload is not None and init_velocity is not None:
        scene.set_gaussians(gaussians,
                            os.path.join(os.path.dirname(os.path.dirname(args.dy_reload)), 'checkpoint/gaussians.ply'))
    else:
        scene.set_gaussians(gaussians, os.path.join(cfg.CHECKPOINTS_ROOT, f'{static_name}/point_cloud.ply'))

    xyz_all = gaussians.get_xyz

    logger.info(f"Got {colored(xyz_all.shape[0], 'yellow', attrs=['bold'])} points in all")
    if args.dy_reload is not None and init_velocity is not None:
        xyz = torch.load(os.path.join(os.path.dirname(os.path.dirname(args.dy_reload)), 'checkpoint/anchors.pt')).cuda()
    else:
        try:
            xyz = uniform_sampling(xyz_all, voxel_size=0.01)
            xyz = xyz[random.sample(range(xyz.shape[0]), cfg_stage.N_SAMPLE), :]
        except:
            xyz = uniform_sampling(xyz_all, voxel_size=0.001)
            xyz = xyz[random.sample(range(xyz.shape[0]), cfg_stage.N_SAMPLE), :]

    simulator = build_simulator(
        cfg_stage,
        xyz=xyz,
        data=cfg.DATA,
        init_velocity=init_velocity,
        load_g=load_g,
    )
    simulator = simulator.cuda()
    simulator.set_all_particle(xyz_all)

    if args.dy_reload is not None and init_velocity is not None:
        with open(os.path.join(os.path.dirname(os.path.dirname(args.dy_reload)), 'checkpoint/dy_n_step.json'),
                  'r') as file:
            load_dy_n_step = json.load(file)
        print("")
        logger.warning(f"set dynamic n_step from {simulator.n_step} to: {load_dy_n_step}")
        simulator.n_step = load_dy_n_step

    gaussians.training_setup_dynamic(cfg_stage.GAUS_CFG)

    return simulator, gaussians


def train_step(args, cfg, cfg_stage, stage, max_frame, simulator, scene: Scene, gaussians):
    ema_loss_for_log = 0.0
    background = torch.tensor(scene.dataset.bg, dtype=torch.float32, device="cuda")
    simulator.train()
    train_bar = etqdm(range(1, cfg_stage.ITERATIONS + 1))
    for iteration in train_bar:

        if iteration in cfg_stage.N_STEP_ITER:
            if simulator.n_step + cfg_stage.STEP_ADD <= cfg_stage.MAX_N_STEP:
                print("")
                logger.warning(
                    f"set dynamic n_step from {simulator.n_step} to: {simulator.n_step + cfg_stage.STEP_ADD}")
                simulator.n_step = simulator.n_step + cfg_stage.STEP_ADD

        xyz_all = simulator.init_xyz_all.detach().clone()
        xyz = simulator.init_xyz.detach().clone()
        v = simulator.init_v.detach().clone()
        gaussians._xyz = xyz_all

        for frame_id in range(max_frame):
            viewpoint_stack = scene.getTrainCameras(frame_id).copy()
            cam_id = random.randint(1, cfg.DATA.N_CAM) - 1

            if gaussians.optimizer is not None:
                gaussians.optimizer.zero_grad(set_to_none=True)

            if frame_id == 0:
                xyz_all = torch.sum(xyz[simulator.intrp_index] * simulator.intrp_coef.unsqueeze(-1), dim=1)
                gaussians._xyz = xyz_all

                loss, _ = render_step(args=args,
                                      cfg_stage=cfg_stage,
                                      iteration=iteration,
                                      viewpoint_cam=viewpoint_stack[cam_id],
                                      frame_id=frame_id,
                                      gaussians=gaussians,
                                      background=background,
                                      stage=stage,
                                      optim=False)
            else:
                simulator.optimizer.zero_grad(set_to_none=True)

                is_nan = True
                while is_nan:
                    xyz_all_o, xyz_o, v_o, is_nan = simulator(xyz_all, xyz, v, frame_id)
                    if is_nan:
                        print("")
                        logger.warning(
                            f"set {stage} n_step from {simulator.n_step} to: {simulator.n_step + cfg_stage.STEP_ADD}")
                        simulator.n_step = simulator.n_step + cfg_stage.STEP_ADD
                        assert simulator.n_step <= cfg_stage.MAX_N_STEP, f"Train dynamic failed!"
                    else:
                        gaussians._xyz = xyz_all_o
                        xyz = xyz_o.detach().clone()
                        v = v_o.detach().clone()
                        xyz_all = xyz_all_o.detach().clone()

                        loss, _ = render_step(args=args,
                                              cfg_stage=cfg_stage,
                                              iteration=iteration,
                                              viewpoint_cam=viewpoint_stack[cam_id],
                                              frame_id=frame_id,
                                              gaussians=gaussians,
                                              background=background,
                                              stage=stage,
                                              xyz_all=xyz_all_o,
                                              xyz=xyz_o.detach().clone())

                clip_gradient(simulator.optimizer, 1.0, 2)
                simulator.optimizer.step()
                simulator.optimizer.zero_grad(set_to_none=True)
                if simulator.scheduler is not None:
                    simulator.scheduler.step()

            if gaussians.optimizer is not None and frame_id == 0:
                clip_gradient(gaussians.optimizer, 1.0, 2)
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()

            with torch.no_grad():
                ema_loss_for_log = 0.4 * loss + 0.6 * ema_loss_for_log
                train_bar.set_description(f"{bar_perfixes[stage]} frame: {frame_id}/{max_frame-1} "
                                          f"cam: {cam_id}/{cfg.DATA.N_CAM-1} "
                                          f"loss: {loss:.{7}f} ema_loss: {ema_loss_for_log:.{7}f}")

                if iteration % cfg_stage.VIZ_INTERVAL == 0:
                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(xyz_all.detach().cpu().numpy())
                    ply_write_dir = os.path.join(scene.exp_path, f'viz_simulator_{stage}')
                    os.makedirs(ply_write_dir, exist_ok=True)
                    o3d.io.write_point_cloud(os.path.join(ply_write_dir, f"{iteration}_{frame_id}.ply"), pcd)

        with torch.no_grad():
            if iteration % cfg_stage.SAVE_INTERVAL == 0:
                print("")
                if simulator.scheduler is not None:
                    recorder.record_checkpoints(simulator, simulator.optimizer, simulator.scheduler, iteration - 1, 1,
                                                f"checkpoints_{stage}")
                else:
                    recorder.record_checkpoints_woscheduler(simulator, simulator.optimizer, iteration - 1, 1,
                                                            f"checkpoints_{stage}")
                if stage == 'dynamic':
                    xyz = simulator.init_xyz.detach().clone()
                    gaussians._xyz = simulator.init_xyz_all.detach().clone()
                    torch.save(xyz.cpu(), os.path.join(scene.exp_path, f'checkpoints_dynamic/checkpoint/anchors.pt'))
                    gaussians.save_ply(os.path.join(scene.exp_path, f'checkpoints_dynamic/checkpoint/gaussians.ply'))

                    with open(os.path.join(scene.exp_path, f'checkpoints_dynamic/checkpoint/dy_n_step.json'),
                              'w') as file:
                        json.dump(simulator.n_step, file, indent=4)

    gaussians._xyz = simulator.init_xyz_all.detach().clone()


def refine_step(args, cfg, cfg_stage, stage, simulator, scene: Scene, gaussians: GaussianModel_isotropic):
    viewpoint_stack = None
    ema_loss_for_log = 0.0

    background = torch.tensor(scene.dataset.bg, dtype=torch.float32, device="cuda")

    xyz = simulator.init_xyz.detach().clone()
    xyz_all = torch.sum(xyz[simulator.intrp_index] * simulator.intrp_coef.unsqueeze(-1), dim=1)
    gaussians._xyz = xyz_all

    train_bar = etqdm(range(1, cfg_stage.ITER_REFINE + 1))
    for iteration in train_bar:
        gaussians.optimizer.zero_grad(set_to_none=True)

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras(0).copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

        # Render
        render_pkg = render(
            viewpoint_cam,
            gaussians,
            background,
            override_color=gaussians.get_color,
            debug=True if (iteration - 1) >= args.debug_from else False,
            compute_cov3D_python=args.compute_cov3D_python,
            convert_SHs_python=args.convert_SHs_python,
        )
        image = render_pkg["render"]

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - cfg_stage.LAMBDA_DSSIM) * Ll1 + cfg_stage.LAMBDA_DSSIM * (1.0 - ssim(image, gt_image))
        loss.backward()

        gaussians.optimizer.step()
        gaussians.optimizer.zero_grad(set_to_none=True)

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                train_bar.set_description(f"REFINE loss: {ema_loss_for_log:.{7}f}")

            if iteration % cfg_stage.REFINE_VIZ_INTERVAL == 0:
                img_list = [
                    image.detach().cpu().permute(1, 2, 0).numpy() * 255,
                    gt_image.detach().cpu().permute(1, 2, 0).numpy() * 255
                ]
                img_list = np.hstack(img_list).astype(np.uint8)
                img_write_dir = os.path.join(scene.exp_path, f'{stage}_refine')
                os.makedirs(img_write_dir, exist_ok=True)
                imageio.imwrite(os.path.join(img_write_dir, f"{iteration}.png"), img_list)


def train_velocity(args, cfg, scene: Scene, recorder: Recorder):
    cfg_velocity = cfg.VELOCITY
    simulator, gaussians = get_simulator(args, cfg, scene, cfg_stage=cfg_velocity, init_velocity=None, load_g=None)
    simulator.trainging_velocity_setup(cfg_velocity.OPTIMIZE)
    if cfg_velocity.ITER_REFINE > 0:
        gaussians.training_setup_refine(cfg_velocity.GAUS_REFINE)
        refine_step(
            args=args,
            cfg=cfg,
            cfg_stage=cfg_velocity,
            stage='velocity',
            simulator=simulator,
            scene=scene,
            gaussians=gaussians,
        )
    gaussians.training_setup_dynamic(cfg_velocity.GAUS_CFG)

    train_step(
        args=args,
        cfg=cfg,
        cfg_stage=cfg_velocity,
        stage='velocity',
        max_frame=scene.dataset.hit_frame,
        simulator=simulator,
        scene=scene,
        gaussians=gaussians,
    )

    recorder.record_checkpoints(simulator, simulator.optimizer, simulator.scheduler, cfg_velocity.ITERATIONS, 1,
                                "checkpoints_velocity")

    if cfg.REAL:
        with open(os.path.join(cfg.CHECKPOINTS_ROOT, f'init_velocity_{cfg.VELOCITY.ITERATIONS}_seq{cfg.DATA.SEQ}.json'),
                  'w') as file:
            json.dump(simulator.init_velocity.detach().cpu().tolist(), file, indent=4)
        if cfg_velocity.OPTIM_G:
            with open(os.path.join(cfg.CHECKPOINTS_ROOT, f'g_{cfg.VELOCITY.ITERATIONS}_seq{cfg.DATA.SEQ}.json'),
                      'w') as file:
                json.dump(simulator.g.detach().cpu().tolist(), file, indent=4)
    else:
        with open(os.path.join(cfg.CHECKPOINTS_ROOT, f'init_velocity_{cfg.VELOCITY.ITERATIONS}.json'), 'w') as file:
            json.dump(simulator.init_velocity.detach().cpu().tolist(), file, indent=4)


def train_dynamic(args, cfg, scene: Scene, recorder: Recorder):
    cfg_dynamic = cfg.DYNAMIC
    if cfg.REAL:
        with open(os.path.join(cfg.CHECKPOINTS_ROOT, f'init_velocity_{cfg.VELOCITY.ITERATIONS}_seq{cfg.DATA.SEQ}.json'),
                  'r') as file:
            load_velocity = json.load(file)
        if cfg_dynamic.OPTIM_G:
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
                                         cfg_stage=cfg_dynamic,
                                         init_velocity=load_velocity,
                                         load_g=load_g)
    simulator.training_setup(cfg_dynamic.OPTIMIZE)

    if args.dy_reload is None and cfg_dynamic.ITER_REFINE > 0:
        gaussians.training_setup_refine(cfg_dynamic.GAUS_CFG)
        refine_step(
            args=args,
            cfg=cfg,
            cfg_stage=cfg_dynamic,
            stage='dynamic',
            simulator=simulator,
            scene=scene,
            gaussians=gaussians,
        )

    gaussians.training_setup_dynamic(cfg_dynamic.GAUS_CFG)

    train_step(
        args=args,
        cfg=cfg,
        cfg_stage=cfg_dynamic,
        stage='dynamic',
        max_frame=scene.dataset.n_frames,
        simulator=simulator,
        scene=scene,
        gaussians=gaussians,
    )
    if simulator.scheduler is None:
        recorder.record_checkpoints_woscheduler(simulator, simulator.optimizer, cfg_dynamic.ITERATIONS, 1,
                                                "checkpoints_dynamic")
    else:
        recorder.record_checkpoints(simulator, simulator.optimizer, simulator.scheduler, cfg_dynamic.ITERATIONS, 1,
                                    "checkpoints_dynamic")
    gaussians.save_ply(os.path.join(scene.exp_path, 'checkpoints_dynamic/checkpoint/gaussians.ply'))
    torch.save(simulator.init_xyz.detach().clone().cpu(),
               os.path.join(scene.exp_path, f'checkpoints_dynamic/checkpoint/anchors.pt'))

    with open(os.path.join(scene.exp_path, f'checkpoints_dynamic/checkpoint/dy_n_step.json'), 'w') as file:
        json.dump(simulator.n_step, file, indent=4)

    # Evaluation
    if args.eval_cam == -1:
        cam_id = random.randint(1, cfg.DATA.N_CAM) - 1
    else:
        cam_id = args.eval_cam
    logger.info(f"Beginning eval at camera: {cam_id}")

    # Eval Fitting
    eval_fitting(args, cam_id, scene, simulator, gaussians, cfg_dynamic)

    # Eval Prediction
    eval_prediction(args, cam_id, scene, simulator, gaussians, cfg_dynamic)


@torch.no_grad()
def eval_fitting(args, cam_id, scene, simulator, gaussians, cfg_dynamic):
    simulator.eval()

    background = torch.tensor(scene.dataset.bg, dtype=torch.float32, device="cuda")

    # eval fitting
    xyz_all = simulator.init_xyz_all.detach().clone()
    xyz = simulator.init_xyz.detach().clone()
    v = simulator.init_v.detach().clone()
    gaussians._xyz = xyz_all

    xyz_all = torch.sum(xyz[simulator.intrp_index] * simulator.intrp_coef.unsqueeze(-1), dim=1)
    gaussians._xyz = xyz_all

    for frame_id in etqdm(range(scene.dataset.n_frames)):
        viewpoint_cam = scene.getEvalCameras(frame_id, cam_id)
        loss, _ = render_step(args=args,
                              cfg_stage=cfg_dynamic,
                              iteration=-1,
                              viewpoint_cam=viewpoint_cam,
                              frame_id=frame_id,
                              gaussians=gaussians,
                              background=background,
                              stage='eval',
                              optim=False)
        # print(loss)
        pcd = o3d.geometry.PointCloud()
        if args.viz_anchor:
            pcd.points = o3d.utility.Vector3dVector(xyz.detach().cpu().numpy())
        else:
            pcd.points = o3d.utility.Vector3dVector(xyz_all.detach().cpu().numpy())
        ply_write_dir = os.path.join(scene.exp_path, 'evaluations/simulate_fit')
        os.makedirs(ply_write_dir, exist_ok=True)
        o3d.io.write_point_cloud(os.path.join(ply_write_dir, f"fit_{frame_id}.ply"), pcd)

        xyz_all, xyz, v, _ = simulator(xyz_all, xyz, v, frame_id + 1)
        gaussians._xyz = xyz_all

    # exit()
    eval_img_dir = os.path.join(scene.exp_path, 'evaluations/images_fit')
    image_to_video(eval_img_dir,
                   os.path.join(scene.exp_path, 'evaluations', f"fit_cam{cam_id}.mp4"),
                   fps=20 if cfg.REAL else 10)


@torch.no_grad()
def eval_prediction(args, cam_id, scene, simulator, gaussians, cfg_dynamic):
    simulator.eval()

    background = torch.tensor(scene.dataset.bg, dtype=torch.float32, device="cuda")

    eval_freq = cfg.DATA.get('EVAL_FREQ', -1)
    if eval_freq == -1:
        eval_dt = cfg.DATA.EVAL_DT
        simulator.set_dt(dt=eval_dt)
    else:
        simulator.set_dt(freq=cfg.DATA.EVAL_FREQ)

    xyz_all = simulator.init_xyz_all.detach().clone()
    xyz = simulator.init_xyz.detach().clone()
    v = simulator.init_v.detach().clone()
    gaussians._xyz = xyz_all

    xyz_all = torch.sum(xyz[simulator.intrp_index] * simulator.intrp_coef.unsqueeze(-1), dim=1)
    gaussians._xyz = xyz_all

    for frame_id in etqdm(range(cfg.DATA.EVAL_FRAME)):
        viewpoint_cam = scene.getEvalCameras(0, cam_id)
        loss, render_img = render_step(args=args,
                                       cfg_stage=cfg_dynamic,
                                       iteration=-2,
                                       viewpoint_cam=viewpoint_cam,
                                       frame_id=frame_id,
                                       gaussians=gaussians,
                                       background=background,
                                       stage='eval',
                                       optim=False)

        pcd = o3d.geometry.PointCloud()
        if args.viz_anchor:
            pcd.points = o3d.utility.Vector3dVector(xyz.detach().cpu().numpy())
        else:
            pcd.points = o3d.utility.Vector3dVector(xyz_all.detach().cpu().numpy())
        ply_write_dir = os.path.join(scene.exp_path, 'evaluations/simulate_pred')
        os.makedirs(ply_write_dir, exist_ok=True)
        o3d.io.write_point_cloud(os.path.join(ply_write_dir, f"pred_{frame_id}.ply"), pcd)

        xyz_all, xyz, v, _ = simulator(
            xyz_all,
            xyz,
            v,
            frame_id + 1,
            viz=True,
            viz_image=render_img,
            viewpoint_cam=viewpoint_cam,
            viz_force_dir=os.path.join(scene.exp_path, 'evaluations', 'pred_viz'),
        )
        gaussians._xyz = xyz_all

    eval_img_dir = os.path.join(scene.exp_path, 'evaluations/images_pred')
    image_to_video(eval_img_dir, os.path.join(scene.exp_path, 'evaluations', f"pred_cam{cam_id}.mp4"), fps=20)
    image_to_video(os.path.join(scene.exp_path, 'evaluations/pred_viz'),
                   os.path.join(scene.exp_path, 'evaluations', f"viz_cam{cam_id}.mp4"),
                   fps=20)


if __name__ == '__main__':
    exp_time = time()
    arg = config_parser()
    cfg = get_config_merge_default(config_file=arg.cfg, arg=arg)

    logger.warning(f"final args and cfg: \n{format_args_cfg(arg, cfg)}")

    os.environ['CUDA_VISIBLE_DEVICES'] = arg.gpu_id
    import open3d as o3d  # Must import open3d after set CUDA_VISIBLE_DEVICES!

    setup_seed(cfg.SEED)
    recorder = Recorder(arg.exp_id, cfg, rank=0, time_f=exp_time)
    exp_path = f"{recorder.exp_id}_{recorder.timestamp}"

    if not os.path.exists('checkpoints'):
        os.mkdir('checkpoints')

    static_name = 'static_gaussians'
    if not os.path.exists(os.path.join(cfg.CHECKPOINTS_ROOT, static_name)):
        scene = Scene(cfg, exp_path, shuffle=False, load_static=True)
    else:
        scene = Scene(cfg, exp_path, shuffle=False, load_static=False)

    # Static
    if not os.path.exists(os.path.join(cfg.CHECKPOINTS_ROOT, static_name)):
        logger.info("Begin Train Static...")
        train_static(arg, cfg, scene, cfg.CHECKPOINTS_ROOT)

    # Register
    if cfg.REAL:
        from lib.metrics.vgg_loss import VGGLoss, TVLoss
        crit_vgg = VGGLoss().cuda()
        crit_tv = TVLoss(p=2)
        regist_name = f'regist_gaussians_{cfg.DATA.SEQ}'
        if not os.path.exists(os.path.join(cfg.CHECKPOINTS_ROOT, regist_name)):
            scene._getSceneInfo(load_static=False, real_must_img=True)
            register_gaus(arg, cfg, scene, cfg.CHECKPOINTS_ROOT)
            scene._getSceneInfo(load_static=False)

    # Velocity
    if cfg.REAL:
        velocity_file = os.path.join(cfg.CHECKPOINTS_ROOT,
                                     f'init_velocity_{cfg.VELOCITY.ITERATIONS}_seq{cfg.DATA.SEQ}.json')
    else:
        velocity_file = os.path.join(cfg.CHECKPOINTS_ROOT, f'init_velocity_{cfg.VELOCITY.ITERATIONS}.json')
    if not os.path.exists(velocity_file):
        logger.info("Begin Train Velocity...")
        if cfg.REAL:
            scene._getSceneInfo(load_static=False, real_must_img=True)
        train_velocity(arg, cfg, scene, recorder)
        if cfg.REAL:
            scene._getSceneInfo(load_static=False)

    # Dynamic
    train_dynamic(arg, cfg, scene, recorder)
