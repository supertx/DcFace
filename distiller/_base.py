'''
Date: 2024-10-08 15:46:37
LastEditTime: 2025-03-06 10:09:13
Description: 
'''
import os.path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.storage
from torchvision.transforms import functional as F
from tqdm import tqdm
from torch.nn import DataParallel
import time
from torch.utils.data import DataLoader

from dataset.dataset import PreloadFaceDataset


class Distiller(nn.Module):

    def __init__(self, student, teacher, cfg):
        super(Distiller, self).__init__()
        self.student = student
        self.teacher = teacher
        preload_batch = cfg.PRELOAD_BATCH
        self.cfg = cfg
        mix_flip_face_dataset = PreloadFaceDataset(cfg.DATASET.DATA_DIR)
        loader = DataLoader(mix_flip_face_dataset,
                            batch_size=preload_batch,
                            shuffle=False,
                            num_workers=4)

        self.teacher_logit = torch.empty((len(mix_flip_face_dataset), 512))
        self.teacher_flip_logit = torch.empty(
            (len(mix_flip_face_dataset), 512))
        self.teacher.eval()
        self.teacher.cuda()
        model = DataParallel(self.teacher)
        time.sleep(0.5)
        # 初始化时将所有的教师的logit都保存下来，防止之后每一个epoch都重复计算
        # TODO 将教师的logit保存到文件中，下次直接读取
        if os.path.isfile(
                    f"./model/data/{cfg.DISTILLER.TEACHER}_teacher_logit.pth")  \
            and os.path.isfile(
                    f"./model/data/{cfg.DISTILLER.TEACHER}_teacher_flip_logit.pth"):
            self.teacher_logit = torch.load(
                f"./model/data/{cfg.DISTILLER.TEACHER}_teacher_logit.pth")
            self.teacher_flip_logit = torch.load(
                f"./model/data/{cfg.DISTILLER.TEACHER}_teacher_flip_logit.pth")
        else:
            t = tqdm(total=len(loader),
                     desc="Preload teacher logit",
                     ncols=100)
            with torch.no_grad():
                for i, (img, flip_img) in enumerate(loader):
                    # 将512个数据拼成一个tensor，然后一次性计算
                    # batch = torch.stack([dataset[i + j][1] for j in
                    #                      range(preload_batch)]).cuda() if self.teacher.cuda() else torch.tensor(
                    #     [dataset[i + j][1] for j in range(preload_batch)])

                    self.teacher_logit[i * preload_batch:min(
                        (i + 1) *
                        preload_batch, self.teacher_logit.shape[0])] = model(
                            img.cuda()).cpu()
                    self.teacher_flip_logit[i * preload_batch:min(
                        (i + 1) *
                        preload_batch, self.teacher_logit.shape[0])] = model(
                            flip_img.cuda()).cpu()
                    t.update(1)
                torch.save(
                    self.teacher_logit,
                    f"./model/data/{cfg.DISTILLER.TEACHER}_teacher_logit.pth")
                torch.save(
                    self.teacher_flip_logit,
                    f"./model/data/{cfg.DISTILLER.TEACHER}_teacher_flip_logit.pth"
                )
            t.close()
        self.teacher.cpu()
        self.teacher_logit.requires_grad = False
        self.teacher_flip_logit.requires_grad = False
        torch.cuda.empty_cache()

    # 设置模型的训练状态
    def train(self, mode=True):
        # teacher as eval mode by default
        if not isinstance(mode, bool):
            raise ValueError("training mode is expected to be boolean")
        self.training = mode
        for module in self.children():
            module.train(mode)
        self.teacher.eval()
        return self

    def get_learnable_parameters(self):
        # if the method introduces extra parameters, re-impl this function
        return [v for k, v in self.student.named_parameters()]

    def get_extra_parameters(self):
        # calculate the extra parameters introduced by the distiller
        return 0

    def forward_train(self, **kwargs):
        # training function for the distillation method
        raise NotImplementedError()

    def forward_test(self, image):
        return self.student(image)[0]

    def forward(self, **kwargs):
        if self.training:
            return self.forward_train(**kwargs)
        return self.forward_test(kwargs["image"])


class Distiller_Origin(nn.Module):

    def __init__(self, student, teacher):
        super(Distiller_Origin, self).__init__()
        self.student = student
        self.teacher = teacher

    # 设置模型的训练状态
    def train(self, mode=True):
        # teacher as eval mode by default
        if not isinstance(mode, bool):
            raise ValueError("training mode is expected to be boolean")
        self.training = mode
        for module in self.children():
            module.train(mode)
        self.teacher.eval()
        return self

    def get_learnable_parameters(self):
        # if the method introduces extra parameters, re-impl this function
        return [v for k, v in self.student.named_parameters()]

    def get_extra_parameters(self):
        # calculate the extra parameters introduced by the distiller
        return 0

    def forward_train(self, **kwargs):
        # training function for the distillation method
        raise NotImplementedError()

    def forward_test(self, image):
        return self.student(image)[0]

    def forward(self, **kwargs):
        if self.training:
            return self.forward_train(**kwargs)
        return self.forward_test(kwargs["image"])


class Vanilla(nn.Module):

    def __init__(self, student):
        super(Vanilla, self).__init__()
        self.student = student

    def get_learnable_parameters(self):
        return [v for k, v in self.student.named_parameters()]

    def forward_train(self, image, target, **kwargs):
        logits_student, _ = self.student(image)
        loss = F.cross_entropy(logits_student, target)
        return logits_student, {"ce": loss}

    def forward(self, **kwargs):
        if self.training:
            return self.forward_train(**kwargs)
        return self.forward_test(kwargs["image"])

    def forward_test(self, image):
        return self.student(image)[0]
