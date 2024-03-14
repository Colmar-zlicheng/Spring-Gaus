import os
import cv2
import json
import math
import numpy as np
from copy import deepcopy
from typing import Any, NamedTuple
from PIL import Image
from termcolor import colored
from lib.utils.etqdm import etqdm
from lib.utils.logger import logger
from lib.utils.builder import DATASET
from lib.utils.read_cameras import read_cameras_binary, read_images_binary


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
class Real_Capture():

    def __init__(self, cfg) -> None:
        self.name = type(self).__name__
        self.cfg = cfg

        self.obj_name = cfg.OBJ_NAME
        self.data_root = cfg.DATA_ROOT
        self.static_dir = os.path.join(self.data_root, 'static')
        self.dynamic_dir = os.path.join(self.data_root, 'dynamic')

        self.seq = cfg.SEQ
        self.H = cfg.H
        self.W = cfg.W
        self.H_S = cfg.H_S
        self.W_S = cfg.W_S
        self.img_is_mask = cfg.get('IMG_IS_MASK', False)
        self.num_views = cfg.N_CAM
        self.all_frames = cfg.FRAME_ALL

        self.white_bkg = False
        self.bg = np.array([0, 0, 0])

        logger.info(f"{self.name}: {colored(self.obj_name, 'yellow', attrs=['bold'])} use seq: {self.seq}")

    def get_colmap_info(self):
        camdata = read_cameras_binary(os.path.join(self.static_dir, f'colmap/{self.obj_name}/cameras.bin'))
        imdata = read_images_binary(os.path.join(self.static_dir, f'colmap/{self.obj_name}/images.bin'))

        imdata = sorted(imdata.items(), reverse=False)

        K_static = np.array([[camdata[1].params[0], 0, camdata[1].params[2]],
                             [0, camdata[1].params[1], camdata[1].params[3]], [0, 0, 1]])

        static_cams = []
        self.static_nimgs = len(imdata)

        for i, (_, im) in enumerate(etqdm(imdata)):
            R = im.qvec2rotmat()
            t = im.tvec

            image_path = os.path.join(self.static_dir, 'images', self.obj_name, im.name)
            image = Image.open(image_path)
            mask_path = image_path.replace('.JPG', '.png').replace('/images/', '/masks/')
            mask = Image.open(mask_path)
            if mask.size != image.size:
                image = image.resize(mask.size)
            im_data = np.array(image)
            height, width = im_data.shape[:2]
            mask = np.array(mask)[:, :, np.newaxis] / 255.0
            arr = (im_data / 255.0) * mask + self.bg * (1 - mask)
            image = Image.fromarray(np.array(arr * 255.0, dtype=np.byte), "RGB")

            static_cams.append(
                CameraInfo(uid=i,
                           R=np.transpose(R),
                           T=np.array(t),
                           FovY=focal2fov(camdata[1].params[1], height),
                           FovX=focal2fov(camdata[1].params[0], width),
                           image=image,
                           image_path=image_path,
                           image_name=im.name,
                           width=width,
                           height=height))

        logger.info(f"{self.name}: {self.obj_name}, Got {colored(self.static_nimgs, 'yellow', attrs=['bold'])} images")

        return static_cams

    def get_cam_info(self, img_is_mask=None):
        if img_is_mask is None:
            img_is_mask = self.img_is_mask

        camdata = read_cameras_binary(os.path.join(self.static_dir, f'colmap/{self.obj_name}/cameras.bin'))
        cam_name = ['C0733', 'C0787', 'C0801']

        K = np.array([
            [camdata[1].params[0] * self.W / self.W_S, 0, self.W / 2],
            [0, camdata[1].params[1] * self.H / self.H_S, self.H / 2],
            [0, 0, 1],
        ])

        FovY = focal2fov(K[0][0], self.H)
        FovX = focal2fov(K[1][1], self.W)

        with open(os.path.join(self.dynamic_dir, 'cameras_calib.json'), 'r') as f:
            cam_calib = json.load(f)
        with open(os.path.join(self.dynamic_dir, 'sync.json'), 'r') as ff:
            sync = json.load(ff)

        with open(os.path.join(self.dynamic_dir, f'sequences/{self.obj_name}/{self.seq}.json'), 'r') as fff:
            seq_info = json.load(fff)

        logger.info('Reading all data ...')
        self.hit_frame = seq_info['hit_frame']
        self.n_frames = len(seq_info[cam_name[0]])
        logger.info(f"{self.name}: {self.obj_name}, Got {colored(self.n_frames, 'yellow', attrs=['bold'])} frames")
        cam_infos_all = []

        for frame_id in etqdm(range(self.n_frames)):
            cam_infos = []
            for cam_id, camera in enumerate(cam_name):
                rvecs = cam_calib[camera]['rvecs']
                tvecs = cam_calib[camera]['tvecs']
                rot_mat, _ = cv2.Rodrigues(np.array(rvecs))
                R = np.transpose(rot_mat)

                image_path = os.path.join(self.dynamic_dir, 'videos_images', camera, seq_info[camera][frame_id])
                mask_path = image_path.replace('/videos_images/', '/videos_masks/').replace('.jpg', '.png')

                if img_is_mask:
                    mask = Image.open(mask_path)
                    image = Image.fromarray(np.repeat(np.array(mask)[:, :, np.newaxis], 3, axis=-1), "RGB")
                else:
                    image = Image.open(image_path)
                    im_data = np.array(image)
                    mask = Image.open(mask_path)
                    mask = np.array(mask)[:, :, np.newaxis] / 255.0
                    arr = (im_data / 255.0) * mask + self.bg * (1 - mask)
                    image = Image.fromarray(np.array(arr * 255.0, dtype=np.byte), "RGB")

                cam_infos.append(
                    CameraInfo(uid=cam_id,
                               R=R,
                               T=np.array(tvecs).reshape(3),
                               FovY=FovY,
                               FovX=FovX,
                               image=image,
                               image_path=image_path,
                               image_name=seq_info[camera][frame_id],
                               width=self.W,
                               height=self.H))

            cam_infos_all.append(cam_infos)

        return cam_infos_all
