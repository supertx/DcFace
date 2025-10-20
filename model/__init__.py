'''
Author: supermantx
Date: 2024-10-08 15:48:44
LastEditTime: 2024-12-12 17:01:13
Description: 
'''
import torch

from .iresnet import iresnet18, iresnet34, iresnet50, iresnet100
from .mobilefacenet import get_mbf
from .parameternet import parameternet_fr


def get_model(model_name: str, cfg=None, **kwargs) -> torch.nn.Module:
    if model_name.lower() == "resnet18":
        return iresnet18(fp16=cfg.SOLVER.FP16, **kwargs)
    elif model_name.lower() == "resnet34":
        return iresnet34(fp16=cfg.SOLVER.FP16, **kwargs)
    elif model_name.lower() == "resnet50":
        return iresnet50(fp16=cfg.SOLVER.FP16, **kwargs)
    elif model_name.lower() == "resnet100":
        return iresnet100(fp16=cfg.SOLVER.FP16, **kwargs)
    elif model_name.lower() == "mobilenetv2":
        if cfg:
            return get_mbf(cfg.SOLVER.FP16,
                           num_features=512,
                           blocks=(1, 4, 6, 2),
                           scale=2)
        return get_mbf(True, 512, blocks=(1, 4, 6, 2), scale=2, **kwargs)
    elif model_name.lower() == "parameternet":
        return parameternet_fr(fp16=cfg.SOLVER.FP16, **kwargs)
    else:
        raise ValueError(f"Unknown model: {model_name}")
