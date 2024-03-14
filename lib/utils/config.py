from argparse import Namespace
from copy import deepcopy

from yacs.config import CfgNode as CN
from lib.utils.logger import logger


class CN_R(CN):

    def recursive_cfg_update(self):

        for k, v in self.items():
            if isinstance(v, list):
                for i, v_ in enumerate(v):
                    if isinstance(v_, dict):
                        new_v = CN_R(v_, new_allowed=True)
                        v[i] = new_v.recursive_cfg_update()
            elif isinstance(v, CN) or issubclass(type(v), CN):
                new_v = CN_R(v, new_allowed=True)
                self[k] = new_v.recursive_cfg_update()
        self.freeze()
        return self

    def dump(self, *args, **kwargs):

        def change_back(cfg: CN_R) -> dict:
            for k, v in cfg.items():
                if isinstance(v, list):
                    for i, v_ in enumerate(v):
                        if isinstance(v_, CN_R):
                            new_v = change_back(v_)
                            v[i] = new_v
                elif isinstance(v, CN_R):
                    new_v = change_back(v)
                    cfg[k] = new_v
            return dict(cfg)

        cfg = change_back(deepcopy(self))
        return CN(cfg).dump(*args, **kwargs)



def get_config_merge_default(config_file: str, arg: Namespace = None) -> CN:
    tmp_cfg = CN(new_allowed=True)
    tmp_cfg.merge_from_file(config_file)
    default_path = tmp_cfg.DEFAULT
    del tmp_cfg

    cfg = CN(new_allowed=True)
    cfg.merge_from_file(default_path)
    cfg.merge_from_file(config_file)

    if arg is not None:
        if arg.v_reload is not None:
            logger.warning(f"cfg VELOCITY's pretrained {cfg.VELOCITY.PRETRAINED} reset to arg.v_reload: {arg.v_reload}")
            cfg.VELOCITY.PRETRAINED = arg.v_reload

        if arg.dy_reload is not None:
            logger.warning(f"cfg DYNAMIC's pretrained {cfg.DYNAMIC.PRETRAINED} reset to arg.dy_reload: {arg.dy_reload}")
            cfg.DYNAMIC.PRETRAINED = arg.dy_reload

        if arg.eval:
            logger.warning(f"Do evaluation!")
            cfg.DYNAMIC.ITERATIONS = 0
        
        if arg.boundary_condition is not None:
            logger.warning(f"cfg DATA's BC {cfg.DATA.BC} reset to arg.boundary_condition: {arg.boundary_condition}")
            cfg.DATA.BC = arg.boundary_condition
        
        if arg.global_k is not None:
            logger.warning(f"cfg DATA's GLOBAL_K {cfg.DATA.GLOBAL_K} reset to arg.global_k: {arg.global_k}")
            cfg.DATA.GLOBAL_K = arg.global_k

    cfg = CN_R(cfg, new_allowed=True)
    cfg.recursive_cfg_update()
    # cfg.freeze()
    return cfg