
import torch
from torch import nn
from queue import Queue
from torch.nn import functional as F
from torch.nn import DataParallel
import torch.distributed as dist


class MarginLoss(nn.Module):

    def __init__(self):
        super(MarginLoss, self).__init__()

    def forward(self, logits, centers, labels):
        # print(labels)
        # print(labels.shape)
        # print(centers.shape)
        # print(logits.shape)
        X = F.linear(F.normalize(logits, eps=1e-7),
                     F.normalize(centers, eps=1e-7),
                     bias=None).cuda()
        res = F.cross_entropy(X, labels)
        return res


class TLoss(nn.Module):

    def __init__(self, cfg):
        super(TLoss, self).__init__()
        self.cfg = cfg
        self.center_queue = Queue(maxsize=cfg.KD.LOSS.QUEUE_SIZE)
        self.t_weight = cfg.KD.LOSS.T_WEIGHT
        self.MarginLoss = DataParallel(MarginLoss().cuda())

    def forward(self, logits, centers):
        l = len(self.center_queue.queue)
        for center in centers:
            if self.center_queue.full():
                self.center_queue.get()
            self.center_queue.put(center.cpu())
            if l < self.cfg.KD.LOSS.QUEUE_SIZE - self.cfg.DATASET.BATCH_SIZE:
                labels = torch.arange(0, len(logits)) + l
            else:
                labels = torch.arange(
                    self.cfg.KD.LOSS.QUEUE_SIZE - len(logits),
                    self.cfg.KD.LOSS.QUEUE_SIZE)
        labels = labels.cuda().long()
        centers = torch.stack(list(self.center_queue.queue)).cuda()
        return self.t_weight * self.MarginLoss(logits,
                                               torch.cat(
                                                   (centers, centers)), labels)


class TLoss_v2(nn.Module):

    def __init__(self, cfg):
        super(TLoss_v2, self).__init__()
        self.cfg = cfg
        self.center_queue = Queue(maxsize=cfg.KD.LOSS.QUEUE_SIZE)
        self.t_weight = cfg.KD.LOSS.T_WEIGHT
        self.t = cfg.KD.LOSS.T_TEMPERATURE

    def forward(self, features, teacher_features):
        for feature in reversed(teacher_features):
            if self.center_queue.full():
                self.center_queue.get()
            self.center_queue.put(feature.cpu())
        teacher_centers = torch.stack(list(reversed(
            self.center_queue.queue))).to(features.device)
        teacher_features_n = F.normalize(teacher_centers, eps=1e-7)
        soft_target = torch.matmul(
            teacher_features_n[:self.cfg.DATASET.BATCH_SIZE],
            teacher_features_n.t())
        soft_target /= self.t
        soft_target = F.softmax(soft_target, dim=1)

        similarity = F.linear(F.normalize(features, eps=1e-7),
                              teacher_features_n,
                              bias=None) / 2 + 0.5
        return self.t_weight * F.cross_entropy(similarity, soft_target)


def log_loss(similarity, soft_target):
    loss = -soft_target * torch.log(similarity)
    return loss.sum(dim=1).mean()


class TLoss_v3(nn.Module):

    def __init__(self, cfg):
        super(TLoss_v3, self).__init__()
        self.cfg = cfg
        self.center_queue = Queue(maxsize=cfg.KD.LOSS.QUEUE_SIZE)
        self.t_weight = cfg.KD.LOSS.T_WEIGHT
        self.t = cfg.KD.LOSS.T_TEMPERATURE

    def forward(self, features, teacher_features):
        for feature in reversed(teacher_features):
            if self.center_queue.full():
                self.center_queue.get()
            self.center_queue.put(feature.cpu())
        teacher_centers = torch.stack(list(reversed(
            self.center_queue.queue))).to(features.device)
        teacher_features_n = F.normalize(teacher_centers, eps=1e-7)
        soft_target = torch.matmul(
            teacher_features_n[:self.cfg.DATASET.BATCH_SIZE],
            teacher_features_n.t())

        soft_target = soft_target / 2 + 0.5
        soft_target /= self.t
        soft_target = F.softmax(soft_target, dim=1)
        soft_target /= torch.max(soft_target, dim=1, keepdim=True)[0]

        similarity = F.linear(F.normalize(features, eps=1e-7),
                              teacher_features_n,
                              bias=None) / 2 + 0.5

        return self.t_weight * log_loss(similarity, soft_target)


class TLoss_new(nn.Module):
    """重新写, 使用DDP"""

    def __init__(self, cfg):
        super(TLoss_new, self).__init__()
        self.cfg = cfg
        self.center_queue = Queue(maxsize=cfg.KD.LOSS.QUEUE_SIZE)

    def forward(self, logits, centers):
        gathered_centers = [
            torch.zeros_like(centers) for _ in range(self.cfg.DDP.WORLD_SIZE)
        ]
        dist.all_gather(gathered_centers, centers)
        l = len(self.center_queue.queue)
        for center in gathered_centers:
            if self.center_queue.full():
                self.center_queue.get()
            self.center_queue.put(center.cpu())
            if l < self.cfg.KD.LOSS.QUEUE_SIZE - self.cfg.DATASET.BATCH_SIZE:
                labels = torch.arange(0, len(logits)) + l
            else:
                labels = torch.arange(
                    self.cfg.KD.LOSS.QUEUE_SIZE - len(logits),
                    self.cfg.KD.LOSS.QUEUE_SIZE)
        centers = torch.stack(list(self.center_queue.queue)).cuda()
        labels = labels.cuda().long()
        labels = labels.chunks(self.cfg.DDP.WORLD_SIZE)[dist.get_rank()]
        return self.MarginLoss(logits, centers, labels)
