'''
Date: 2024-11-15 10:23:39
LastEditTime: 2025-04-14 09:58:30
Description: 
'''
from torch.nn import functional as F
import torch
import math


class ArcHead(torch.nn.Module):
    """ ArcFace (https://arxiv.org/pdf/1801.07698v1.pdf):
    """

    def __init__(self, s=64.0, margin=0.5):
        super(ArcHead, self).__init__()
        self.s = s
        self.margin = margin
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.theta = math.cos(math.pi - margin)
        self.sinmm = math.sin(math.pi - margin) * margin
        self.easy_margin = False

    def forward(self, logits: torch.Tensor, labels: torch.Tensor):
        index = torch.where(labels != -1)[0]
        target_logit = logits[index, labels[index].view(-1)]

        with torch.no_grad():
            target_logit.arccos_()
            logits.arccos_()
            final_target_logit = target_logit + self.margin
            logits[index, labels[index].view(-1)] = final_target_logit
            logits.cos_()
        logits = logits * self.s
        return logits


class ArcFace(torch.nn.Module):

    def __init__(self, s=64.0, margin=0.5, cfg=None):
        super(ArcFace, self).__init__()
        self.head = ArcHead(s, margin)
        self.loss = torch.nn.CrossEntropyLoss()
        self.fc = torch.nn.Parameter(
            torch.FloatTensor(cfg.DATASET.NUM_CLASSES, 512))
        torch.nn.init.xavier_uniform_(self.fc)

    def forward(self, logits: torch.Tensor, labels: torch.Tensor):
        logit = F.linear(F.normalize(logits, dim=1),
                         F.normalize(self.fc, dim=1),
                         bias=None)

        logit = logit.clamp(min=-1, max=1)
        logit = self.head(logit, labels)
        loss = self.loss(logit, labels)
        return loss
