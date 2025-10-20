'''
Date: 2024-10-08 15:46:37
LastEditTime: 2025-04-14 09:28:03
Description: 
'''

from distiller.kl import KL
from distiller.KD_TLoss import KD_TLoss
from distiller.fcd import FCD
from .TLoss import *
from distiller.arc_face import ArcFace


def get_distiller(cfg, student, teacher):
    if cfg.DISTILLER.CLASS == "kl":
        return KL(student, teacher, cfg)
    elif cfg.DISTILLER.CLASS == "KD_TLoss":
        return KD_TLoss(student, teacher, cfg)
    elif cfg.DISTILLER.CLASS == "fcd":
        return FCD(student, teacher, cfg)
    else:
        raise ValueError(f"Unknown distiller: {cfg.DISTILLER.NAME}")


def get_arcface(cfg):
    return ArcFace(cfg=cfg)


def get_TLoss(cfg):
    if cfg.DISTILLER.TLoss == 'v1':
        return TLoss(cfg)
    elif cfg.DISTILLER.TLoss == 'v2':
        return TLoss_v2(cfg)
    elif cfg.DISTILLER.TLoss == 'v3':
        return TLoss_v3(cfg)
    else:
        return None
