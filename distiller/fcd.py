'''
Date: 2024-10-08 15:46:38
LastEditTime: 2025-04-14 09:03:47
Description: 
'''
from torch import nn
import torch
import torch.nn.functional as F

from ._base import Distiller


def normalize(logit):
    mean = logit.mean(dim=-1, keepdims=True)
    stdv = logit.std(dim=-1, keepdims=True)
    return (logit - mean) / (1e-7 + stdv)


def fcd_loss(logits_student, logits_teacher):
    # 将特征归一化
    logits_student_norm = normalize(logits_student)
    logits_teacher_norm = normalize(logits_teacher)
    # 计算l2距离
    loss_fcd = F.mse_loss(logits_student_norm,
                          logits_teacher_norm,
                          reduction="mean")
    return loss_fcd


class FCD(Distiller):

    def __init__(self, student, teacher, cfg):
        super().__init__(student, teacher, cfg)
        self.fcd_loss_weight = cfg.KD.LOSS.FCD_WEIGHT

    def forward_train(self,
                      index,
                      image,
                      flip_flag=None,
                      epoch=None,
                      **kwargs):
        logits_student = self.student(image)
        logits_teacher = torch.stack([
            self.teacher_logit[i] if not flag else self.teacher_flip_logit[i]
            for i, flag in zip(index, flip_flag)
        ]).cuda()
        # losses
        loss_fcd = self.fcd_loss_weight * fcd_loss(logits_student,
                                                   logits_teacher)

        if epoch and epoch > self.cfg.KD.EMA_START:
            temp = logits_student.detach().cpu()
            for j, i in enumerate(index.cpu()):
                self.teacher_logit[
                    i] = self.cfg.KD.EMA_ALPHA * self.teacher_logit[i] + (
                        1 - self.cfg.KD.EMA_ALPHA) * temp[j]
                self.teacher_flip_logit[
                    i] = self.cfg.KD.EMA_ALPHA * self.teacher_flip_logit[i] + (
                        1 - self.cfg.KD.EMA_ALPHA) * temp[j]
        return logits_teacher, logits_student, loss_fcd.unsqueeze(0)
