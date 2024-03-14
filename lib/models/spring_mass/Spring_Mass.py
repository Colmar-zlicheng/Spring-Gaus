import os
import torch
import torch.nn as nn
import numpy as np

from termcolor import colored
from copy import deepcopy
from pytorch3d.ops import knn_points
from lib.utils.builder import SIMULATOR
from lib.utils.logger import logger
from lib.utils.misc import param_size
from lib.utils.net_utils import init_weights


@SIMULATOR.register_module()
class Spring_Mass(nn.Module):

    def __init__(self, cfg, xyz: torch.Tensor, init_velocity=None, load_g=None) -> None:
        super().__init__()
        self.name = type(self).__name__
        self.cfg = cfg

        self.eps = 1e-14
        self.edge = 1e-6
        self.k_neighbors = cfg.K_NEIGHBORS
        self.k_binding = cfg.K_BINDING
        self.n_step = cfg.N_STEP
        freq = cfg.DATA.get('FREQ', -1)
        if freq == -1:
            self.dt = cfg.DATA.DT
        else:
            self.dt = 1 / cfg.DATA.FREQ

        self.bc = cfg.DATA.BC
        if self.bc[0][1][1] == 1 or self.bc[0][1][1] == -1:
            self.ground_axis = 1
            self.free_axis = [0, 2]
            self.g_f = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32)
            if self.bc[0][1][1] == -1:
                self.inverse_axis = True
            else:
                self.inverse_axis = False
        elif self.bc[0][1][2] == 1 or self.bc[0][1][2] == -1:
            self.ground_axis = 2
            self.free_axis = [0, 1]
            self.g_f = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32)
            if self.bc[0][1][2] == -1:
                self.inverse_axis = True
            else:
                self.inverse_axis = False
        else:
            raise ValueError()
        self.ground = self.bc[0][0][self.ground_axis]

        fitting = cfg.get('FITTING', True)
        self.stretch_ratios = cfg.get('STRETCH_RATIOS', 0.0)  # 0.05
        self.ratio_factor = cfg.get('RATIO_FACTOR', 1.0)
        self.damping = cfg.get('DAMPING', False)
        self.fix_mass = cfg.get('FIX_MASS', False)
        self.fix_damp = cfg.get('FIX_DAMP', False)
        self.optim_fixed_damp = cfg.get('OPTIM_FIXED_DAMP', False)
        self.fix_k = cfg.get('FIX_K', False)
        self.unlinear_foce = cfg.get('UNLINEAR_FORCE', True)
        self.power = cfg.get('POWER', 0.5)
        self.optim_g = cfg.get('OPTIM_G', False)
        self.soft_k = cfg.get('SOFT_K', True)
        self.spring_bc = cfg.get('SPRING_BC', False)
        self.single_k = cfg.get('SINGLE_K', False)
        if self.optim_g:
            if load_g is not None:
                self.g = nn.Parameter(torch.tensor(load_g, dtype=torch.float32), requires_grad=True)
            else:
                self.g = nn.Parameter(torch.tensor(cfg.G, dtype=torch.float32), requires_grad=True)
        else:
            if load_g is not None:
                self.g = torch.tensor(load_g, dtype=torch.float32)
            else:
                self.g = torch.tensor(cfg.G, dtype=torch.float32)

        self.initialize(xyz)

        if init_velocity is None:
            self.init_velocity = nn.Parameter(torch.tensor(cfg.INIT_VELOCITY, dtype=torch.float32),
                                              requires_grad=fitting)
            self.stage = 'velocity'
        else:
            self.stage = 'dynamic'
            self.optim_dy_velocity = cfg.get('OPTIM_DY_VELOCITY', False)
            if self.optim_dy_velocity:
                self.init_velocity = nn.Parameter(torch.tensor(init_velocity, dtype=torch.float32),
                                                  requires_grad=fitting)
            else:
                self.init_velocity = torch.tensor(init_velocity, dtype=torch.float32)

        if self.fix_k:
            if self.single_k:
                self.global_k = nn.Parameter(torch.log10(torch.tensor(cfg.DATA.GLOBAL_K, dtype=torch.float32)),
                                             requires_grad=fitting)
            else:
                self.global_k = nn.Parameter(torch.log10(torch.tensor(cfg.DATA.GLOBAL_K, dtype=torch.float32)) *
                                             torch.ones(self.n_points, dtype=torch.float32),
                                             requires_grad=fitting)

        else:
            self.global_k = nn.Parameter(torch.log10(torch.tensor(cfg.DATA.GLOBAL_K, dtype=torch.float32)) *
                                         torch.ones_like(self.origin_len, dtype=torch.float32),
                                         requires_grad=fitting)
        if self.fix_mass:
            self.global_m = nn.Parameter(torch.log10(torch.tensor(cfg.DATA.GLOBAL_M, dtype=torch.float32)) *
                                         torch.ones(self.n_points, dtype=torch.float32),
                                         requires_grad=False)
        else:
            self.global_m = nn.Parameter(torch.log10(torch.tensor(cfg.DATA.GLOBAL_M, dtype=torch.float32)) *
                                         torch.ones(self.n_points, dtype=torch.float32),
                                         requires_grad=fitting)

        if self.damping:
            if self.fix_damp:
                if self.optim_fixed_damp:
                    self.damp = nn.Parameter(torch.exp(torch.tensor(cfg.DATA.GLOBAL_DAMP, dtype=torch.float32)),
                                             requires_grad=True)
                else:
                    self.damp = nn.Parameter(torch.log10(torch.tensor(cfg.DATA.GLOBAL_DAMP, dtype=torch.float32)) *
                                             torch.ones_like(self.origin_len, dtype=torch.float32),
                                             requires_grad=False)
            else:
                self.damp = nn.Parameter(torch.log10(torch.tensor(cfg.DATA.GLOBAL_DAMP, dtype=torch.float32)) *
                                         torch.ones_like(self.origin_len, dtype=torch.float32),
                                         requires_grad=fitting)

        self.rebound_k = nn.Parameter(torch.tensor([-1.0], dtype=torch.float32), requires_grad=fitting)
        self.fric_k = nn.Parameter(torch.tensor([-1.0], dtype=torch.float32), requires_grad=fitting)

        if self.soft_k:
            self.n_fix_spring = cfg.get('N_FIX_SPRING', 16)
            self.soft_vector = nn.Parameter(torch.tensor([0.0], dtype=torch.float32), requires_grad=fitting)

        if self.spring_bc:
            self.k_bc = nn.Parameter(torch.log10(torch.tensor(cfg.DATA.GLOBAL_K_BC, dtype=torch.float32)),
                                     requires_grad=fitting)

        logger.info(f"{self.name} has {param_size(self)}M parameters")
        logger.info(f"{self.name} got stage: {self.stage}")
        logger.info(f"{self.name} got k_neighbors: {colored(self.k_neighbors, 'yellow', attrs=['bold'])}")
        logger.info(f"{self.name} got k_binding: {colored(self.k_binding, 'yellow', attrs=['bold'])}")
        logger.info(f"{self.name} got fitting: {fitting}")
        logger.info(f"{self.name} got damping: {self.damping}")
        logger.info(f"{self.name} got fix_mass: {self.fix_mass}")
        logger.info(f"{self.name} got fix_damp: {self.fix_damp}")
        logger.info(f"{self.name} got optim_fixed_damp: {self.optim_fixed_damp}")
        logger.info(f"{self.name} got fix_k: {self.fix_k}")
        logger.info(f"{self.name} got optim_g: {self.optim_g}")
        logger.info(f"{self.name} got soft_k: {self.soft_k}")
        logger.info(f"{self.name} got spring_bc: {self.spring_bc}")
        logger.info(f"{self.name} got single_k: {self.single_k}")
        logger.info(f"{self.name} got unlinear_foce: {self.unlinear_foce} with power: {self.power}")
        logger.info(f"{self.name} got stretch_ratios: {self.stretch_ratios} with ratio_factor: {self.ratio_factor}")
        init_weights(self, pretrained=cfg.PRETRAINED, strict=True)

    def initialize(self, xyz: torch.Tensor):
        self.device = xyz.device
        self.n_points = xyz.shape[0]
        self.init_xyz = xyz.detach().clone()
        self.init_v = torch.zeros_like(self.init_xyz, dtype=torch.float32)

        self.origin_len, self.knn_index, _ = self.knn(self.init_xyz, self.init_xyz, self.k_neighbors, rm_self=True)

        logger.info(f"{self.name} got {colored(self.n_points, 'yellow', attrs=['bold'])} points")

    def trainging_velocity_setup(self, training_args):
        l = [{'params': [self.init_velocity], 'lr': training_args.INIT_VELOCITY_LR, "name": "init_velocity"}]
        if self.optim_g:
            l.append({'params': [self.g], 'lr': training_args.G_LR, "name": "gravity"})

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,
                                                         step_size=training_args.LR_DECAY_STEP,
                                                         gamma=training_args.LR_DECAY_RATE)

    def training_setup(self, training_args):
        l = [{
            'params': [self.global_k],
            'lr': training_args.K_LR,
            "name": "global_k"
        }, {
            'params': [self.rebound_k],
            'lr': training_args.REBOUND_K_LR,
            "name": "rebound_k"
        }, {
            'params': [self.fric_k],
            'lr': training_args.FRIC_K_LR,
            "name": "fric_k"
        }]
        if not self.fix_mass:
            l.append({'params': [self.global_m], 'lr': training_args.M_LR, "name": "global_m"})

        if self.damping and (not self.fix_damp or (self.fix_damp and self.optim_fixed_damp)):
            l.append({'params': [self.damp], 'lr': training_args.DAMPING_LR, "name": "damp"})

        if self.soft_k:
            l.append({'params': [self.soft_vector], 'lr': training_args.SOFT_LR, "name": "soft_vector"})

        if self.spring_bc:
            l.append({'params': [self.k_bc], 'lr': training_args.K_BC_LR, "name": "k_bc"})

        if self.optim_dy_velocity:
            l.append({'params': [self.init_velocity], 'lr': training_args.INIT_VELOCITY_LR, "name": "init_velocity"})

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.scheduler = None

    def knn(self, x: torch.Tensor, ref: torch.Tensor, k, rm_self=False, sqrt_dist=True):
        """
        :param 
            x: [N, 3]
        :return: 
            dist: [N, k]
            knn_dix: [N, k]
            x_neighbor: [N, k, 3]
        """
        if rm_self:
            # self.k_neighbors + 1: because the first one is the point itself
            dist, knn_dix, x_neighbor = knn_points(x.unsqueeze(0), ref.unsqueeze(0), K=k + 1, return_nn=True)
            dist = dist.squeeze(0)[:, 1:]  # [N, k]
            knn_dix = knn_dix.squeeze(0)[:, 1:]  # [N, k]
            x_neighbor = x_neighbor.squeeze(0)[:, 1:]  # [N, k, 3]
        else:
            dist, knn_dix, x_neighbor = knn_points(x.unsqueeze(0), ref.unsqueeze(0), K=k, return_nn=True)
            dist = dist.squeeze(0)  # [N, k]
            knn_dix = knn_dix.squeeze(0)  # [N, k]
            x_neighbor = x_neighbor.squeeze(0)  # [N, k, 3]

        if sqrt_dist:
            return torch.sqrt(dist), knn_dix, x_neighbor
        else:
            return dist, knn_dix, x_neighbor

    def set_all_particle(self, xyz_all: torch.Tensor):
        self.init_xyz_all = xyz_all.detach().clone()
        self.n_all = self.init_xyz_all.shape[0]

        intrp_len, self.intrp_index, _ = self.knn(self.init_xyz_all, self.init_xyz, self.k_binding)
        intrp_coef = 1 / (intrp_len**0.5 + self.eps)
        self.intrp_coef = intrp_coef / (torch.sum(intrp_coef, dim=-1, keepdim=True))  # + self.eps)
        assert self.intrp_coef.sum().int() == self.init_xyz_all.shape[0], \
            "if report an error here, decrease K_BINDING can help"

    def interpolate(self, xyz_all: torch.Tensor, xyz_before: torch.Tensor, delta_xyz: torch.Tensor):
        delta_knn = delta_xyz[self.intrp_index]  # [N, n, 3]
        if self.stage == 'velocity':
            delta_xyz_all = torch.sum(delta_knn * self.intrp_coef.unsqueeze(-1), dim=1)  # [N, 3]
            xyz_all = xyz_all + delta_xyz_all

        elif self.stage == 'dynamic':
            xyz = xyz_before + delta_xyz
            xyz_knn = xyz[self.intrp_index]  # [N, n, 3]
            xyz_all = torch.sum(xyz_knn * self.intrp_coef.unsqueeze(-1), dim=1)  # [N, 3]

        else:
            raise ValueError()

        return xyz_all

    def compute_force(self, xyz, v, K, damp):
        knn_xyz = xyz[self.knn_index]  # [N, k, 3]
        delta_pos = knn_xyz - xyz.unsqueeze(1)  # [N, k, 3]
        curr_len = torch.norm(delta_pos, dim=2)  # [N, k]
        norm_delta_pos = delta_pos / (curr_len.unsqueeze(2) + self.eps)  # [N, k, 3]

        delta_len = (curr_len - self.origin_len)
        delta_len[(delta_len > -self.edge) & (delta_len < self.edge)] = 0.0
        force = (delta_len * K).unsqueeze(2) * norm_delta_pos  # [N, k, 3]

        if self.unlinear_foce and self.power > 0:
            force = force * (1 + torch.abs(curr_len / (self.origin_len + self.eps) - 1).unsqueeze(2))**self.power

        if self.ratio_factor > 1:
            judge = curr_len / (self.origin_len + self.eps) - 1
            mask = torch.where((judge < -self.stretch_ratios) | (judge > self.stretch_ratios))
            force[mask] = self.ratio_factor * force[mask]

        if self.damping:
            knn_v = v[self.knn_index]  # [N, k, 3]
            delta_v = knn_v - v.unsqueeze(1)  # [N, k, 3]
            damp_force = (damp *
                          torch.sum(delta_v * norm_delta_pos, dim=-1)).unsqueeze(-1) * norm_delta_pos  # [N, k, 3]
            force = force + damp_force

        return force.sum(dim=1)

    def apply_bc(self, xyz, v, rebound_k, fric_k):
        # Boundary condition
        if self.inverse_axis:
            v_index = torch.where(xyz[:, self.ground_axis] >= self.ground)[0]
        else:
            v_index = torch.where(xyz[:, self.ground_axis] <= self.ground)[0]

        for axis in self.free_axis:
            v[v_index, axis] = fric_k * v[v_index, axis]

        v[v_index, self.ground_axis] = torch.zeros_like(v_index, device=self.device, dtype=torch.float32)
        xyz[v_index,
            self.ground_axis] = torch.zeros_like(v_index, device=self.device, dtype=torch.float32) + self.ground

        return xyz, v

    def apply_bc_force(self, force, xyz, v, fric_k):
        k_bc = 10**self.k_bc / (self.origin_len.mean() + self.eps)

        if self.inverse_axis:
            f_index = torch.where(xyz[:, self.ground_axis] >= self.ground)[0]
        else:
            f_index = torch.where(xyz[:, self.ground_axis] <= self.ground)[0]

        force_f = -k_bc * (xyz[f_index, self.ground_axis] - self.ground)
        if self.unlinear_foce and self.power > 0:
            force_f = force_f * torch.abs(xyz[f_index, self.ground_axis] - self.ground)**self.power

        force[f_index, self.ground_axis] = force[f_index, self.ground_axis] + force_f
        for axis in self.free_axis:
            frik_force = False
            if frik_force:
                v_direct = v[f_index, axis]
                v_direct = torch.where(
                    v_direct > 0,
                    torch.tensor(1.0).to(self.device),
                    torch.where(v_direct < 0,
                                torch.tensor(-1.0).to(self.device),
                                torch.tensor(0.0).to(self.device)))
                force[f_index, axis] = force[f_index, axis] - force[f_index, axis] * fric_k * v_direct
            else:
                v[f_index, axis] = fric_k * v[f_index, axis]

        return force, xyz, v

    @torch.no_grad()
    def viz_step(self, xyz, v, K, damp, frame_id, **kwargs):
        import cv2
        import imageio
        from lib.utils.transform import SE3_transform, persp_project
        from lib.models.gaus.utils.graphics_utils import fov2focal

        viz_force_dir = kwargs['viz_force_dir']
        viewpoint_cam = kwargs['viewpoint_cam']
        os.makedirs(viz_force_dir, exist_ok=True)

        viz_image = kwargs['viz_image']
        viz_image = (viz_image.permute(1, 2, 0).detach().cpu().numpy() * 255.0).astype(np.uint8)
        frame = viz_image.copy()

        R = viewpoint_cam.R.transpose()
        T = viewpoint_cam.T
        w2c = np.concatenate([R, T[:, None]], axis=-1)
        w2c = np.concatenate([w2c, [[0, 0, 0, 1]]], axis=0)
        # w2c[:3, 1:3] *= -1

        fx = fov2focal(viewpoint_cam.FoVx, viewpoint_cam.image_width)
        fy = fov2focal(viewpoint_cam.FoVy, viewpoint_cam.image_height)
        intrisic = np.array([
            [fx, 0, viewpoint_cam.image_width / 2],
            [0, fy, viewpoint_cam.image_height / 2],
            [0, 0, 1],
        ])

        xyz_c = SE3_transform(xyz.detach().cpu().numpy(), w2c)
        uv = persp_project(xyz_c, intrisic)

        for i in range(uv.shape[0]):
            cx = int(uv[i, 0])
            cy = int(uv[i, 1])
            cv2.circle(frame, (cx, cy), radius=1, thickness=-1, color=np.array([1.0, 0.0, 0.0]) * 255)

        img_list = [frame]
        comb_image = np.hstack(img_list)
        imageio.imwrite(os.path.join(viz_force_dir, f"{frame_id:02}.png"), comb_image)

    def step(self, xyz, v, K, m, rebound_k, fric_k, damp, dt):
        """
        :param
            dt: float
            g: [3]
            xyz: [N, 3]
            knn_xyz: [N, k, 3]
        :medium:
            force: [N, k+1, 3], 1 is gravity
            force_sum: [N, 3]
        :return:
            xyz: [N, 3]
            v: [N, 3]
        """

        # compute_force
        force = self.compute_force(xyz=xyz, v=v, K=K, damp=damp)
        force_sum = force + m.unsqueeze(1) * self.g.unsqueeze(0).to(self.device) * self.g_f.unsqueeze(0).to(
            self.device)  # [N, 3]

        if self.spring_bc:
            force_sum, xyz, v = self.apply_bc_force(force_sum, xyz, v, fric_k)

        # semi-implicit Euler
        v = v + force_sum * dt / m.unsqueeze(1)  # update velocity
        xyz = xyz + v * dt  # update position

        if not self.spring_bc:
            # Boundary condition
            xyz, v = self.apply_bc(xyz, v, rebound_k, fric_k)

        return xyz, v

    def set_dt(self, freq=None, dt=None):
        if (freq is not None and dt is not None) or (freq is None and dt is None):
            assert False
        if freq is not None:
            self.dt = 1 / freq
        if dt is not None:
            self.dt = dt

    def forward(self, xyz_all: torch.Tensor, xyz: torch.Tensor, v: torch.Tensor, frame_id: int, viz=False, **kwargs):
        assert frame_id > 0

        if self.fix_k:
            if self.single_k:
                K = 10**self.global_k.reshape(1, 1).repeat(self.n_points, self.k_neighbors)
            else:
                K = 10**self.global_k.unsqueeze(1).repeat(1, self.k_neighbors)
        else:
            K = 10**self.global_k

        m = 10**self.global_m

        if self.damping:
            if self.optim_fixed_damp:
                damp = torch.log(self.damp) * torch.ones_like(self.origin_len, dtype=torch.float32).to(self.device)
            else:
                damp = 10**self.damp

        rebound_k = torch.sigmoid(self.rebound_k)
        fric_k = torch.clamp(torch.sigmoid(self.fric_k) * 1.2 - 0.1, min=0, max=1)

        K = K / (self.origin_len + self.eps)

        if self.soft_k:
            if self.n_fix_spring == 0:
                soft_learn = torch.arange(self.k_neighbors).to(self.device) / self.k_neighbors
                soft_learn = 2 - torch.pow(torch.exp(torch.nn.functional.softplus(self.soft_vector)), soft_learn)
                k_vector = torch.clamp(soft_learn, 0, 1)
            elif self.n_fix_spring > 0 and self.n_fix_spring < self.k_neighbors:
                soft_learn = torch.arange(
                    (self.k_neighbors - self.n_fix_spring)).to(self.device) / (self.k_neighbors - self.n_fix_spring)
                soft_learn = 2 - torch.pow(torch.exp(torch.nn.functional.softplus(self.soft_vector)), soft_learn)
                soft_learn = torch.clamp(soft_learn, 0, 1)
                k_vector = torch.cat([torch.ones(self.n_fix_spring).to(self.device), soft_learn], dim=0)
            else:
                raise ValueError(f"get invalid n_fix_spring: {self.n_fix_spring}")

            K = K * k_vector.unsqueeze(0)

        if self.damping:
            damp = damp / (self.origin_len + self.eps)
        else:
            damp = None

        xyz_before = deepcopy(xyz)
        v = v + self.init_velocity.unsqueeze(0).to(self.device)

        dt = self.dt / self.n_step

        if viz:
            self.viz_step(xyz, v, K, damp, frame_id, **kwargs)

        for _ in range(self.n_step):
            xyz, v = self.step(xyz=xyz, v=v, K=K, m=m, rebound_k=rebound_k, fric_k=fric_k, damp=damp, dt=dt)
            torch.cuda.empty_cache()

        xyz_all = self.interpolate(xyz_all, xyz_before, delta_xyz=xyz - xyz_before)

        v = v - self.init_velocity.unsqueeze(0).to(self.device)

        is_nan = torch.any(torch.isnan(xyz))
        return xyz_all, xyz, v, is_nan
