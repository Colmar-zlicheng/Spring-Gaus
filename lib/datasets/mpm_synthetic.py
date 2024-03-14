import os
import json
import math
import numpy as np
from copy import deepcopy
from typing import NamedTuple
from PIL import Image
from termcolor import colored
from lib.utils.etqdm import etqdm
from lib.utils.logger import logger
from lib.utils.builder import DATASET


class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int


def focal2fov(focal, pixels):
    return 2 * math.atan(pixels / (2 * focal))


@DATASET.register_module()
class MPM_Synthetic():

    def __init__(self, cfg) -> None:
        self.name = type(self).__name__
        self.cfg = cfg

        self.data_root = cfg.DATA_ROOT
        self.obj_id = cfg.OBJ_NAME
        self.num_views = cfg.N_CAM
        self.n_frames = cfg.N_FRAME
        self.all_frames = cfg.FRAME_ALL
        self.hit_frame = cfg.HIT_FRAME
        self.H = cfg.H
        self.W = cfg.W

        self.obj_name = self.obj_id
        self.data_path = os.path.join(self.data_root, self.obj_name)

        self.white_bkg = False
        self.bg = np.array([0, 0, 0])

        logger.info(f"{self.name}: {colored(self.obj_id, 'yellow', attrs=['bold'])} has "
                    f"{colored(self.num_views, 'yellow', attrs=['bold'])} cameras "
                    f"with {colored(self.n_frames, 'yellow', attrs=['bold'])}/"
                    f"{colored(self.all_frames, 'yellow', attrs=['bold'])} frames")

    def get_cam_info(self):
        with open(os.path.join(self.data_path, 'camera.json'), 'r') as cam_file:
            cameras = json.load(cam_file)

        logger.info('Reading all data ...')

        cam_infos_all = []
        for frame_id in etqdm(range(self.n_frames)):
            cam_infos = []
            for cam_id, camera in enumerate(cameras):

                intrinsic = np.array(camera['K'])
                c2w = deepcopy(np.array(camera['c2w']))
                c2w[:3, 1:3] *= -1
                w2c = np.linalg.inv(c2w)
                R = np.transpose(w2c[:3, :3])  # R is stored transposed due to 'glm' in CUDA code
                T = w2c[:3, 3]

                FovX = focal2fov(intrinsic[0][0], self.W)
                FovY = focal2fov(intrinsic[1][1], self.H)

                cam_name = camera['camera']
                image_path = os.path.join(self.data_path, cam_name, f"{frame_id:03}.png")
                image = Image.open(image_path)
                im_data = np.array(image.convert("RGBA"))
                norm_data = im_data / 255.0
                arr = norm_data[:, :, :3] * norm_data[:, :, 3:4] + self.bg * (1 - norm_data[:, :, 3:4])
                image = Image.fromarray(np.array(arr * 255.0, dtype=np.byte), "RGB")

                cam_infos.append(
                    CameraInfo(uid=cam_id,
                               R=R,
                               T=T,
                               FovY=FovY,
                               FovX=FovX,
                               image=image,
                               image_path=image_path,
                               image_name=f"{cam_name}_{frame_id:03}.png",
                               width=self.W,
                               height=self.H))

            cam_infos_all.append(cam_infos)

        return cam_infos_all
