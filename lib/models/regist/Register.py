import torch
import torch.nn as nn
from copy import deepcopy
from lib.utils.logger import logger
from lib.utils.builder import MODEL
from lib.utils.misc import param_size
from lib.utils.transform import rot6d_to_rotmat, euler_to_quat, quat_to_rot6d, quat_to_rotmat


@MODEL.register_module()
class Register(nn.Module):

    def __init__(self, cfg, init_data):
        super().__init__()
        self.name = type(self).__name__
        self.cfg = cfg

        euler = torch.tensor(init_data.INIT_R, dtype=torch.float32) * torch.pi / 180
        quat = euler_to_quat(euler)
        rot6d = quat_to_rot6d(quat)

        self.r = nn.Parameter(rot6d, requires_grad=True)
        self.t = nn.Parameter(torch.tensor(init_data.INIT_T, dtype=torch.float32), requires_grad=True)
        self.s = nn.Parameter(torch.tensor(init_data.INIT_S, dtype=torch.float32), requires_grad=True)

        logger.info(f"{self.name} has {param_size(self)}M parameters")

    def training_setup(self, training_args):
        l = [{
            'params': [self.r],
            'lr': training_args.R_LR,
            "name": "r"
        }, {
            'params': [self.t],
            'lr': training_args.T_LR,
            "name": "t"
        }, {
            'params': [self.s],
            'lr': training_args.S_LR,
            "name": "s"
        }]
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

    @property
    def get_scale(self):
        return deepcopy(self.s).detach()

    def forward(self, xyz):
        R = rot6d_to_rotmat(self.r)

        origin = torch.mean(xyz, dim=0, keepdim=True)
        xyz = self.s * (xyz - origin)  #+ origin

        xyz = (R @ xyz.transpose(0, 1)).transpose(0, 1) + self.t.unsqueeze(0)

        return xyz
