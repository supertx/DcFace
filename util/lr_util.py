import warnings

import numpy as np
from torch.optim.lr_scheduler import LRScheduler


class StepScaleLR(LRScheduler):
    """
    descend Lr at a fix rate at each stage
    """

    def __init__(self, optimizer, base_lr, stages, gamma, epoch_step):
        warnings.warn(
            "This scheduler is deprecated, please use WarmupCosineAnnealingLR instead.",
            DeprecationWarning)
        super().__init__(optimizer)
        self.base_lr = base_lr
        self.gamma = gamma
        self.stages = [epoch_step * stage for stage in stages]
        self.epoch_step = epoch_step

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn(
                "To get the last learning rate computed by the scheduler, "
                "please use `get_last_lr()`.", UserWarning)
        if self.last_epoch == 0:
            return [group["lr"] for group in self.optimizer.param_groups]
        else:
            time = sum([self.last_epoch >= stage for stage in self.stages])
            return [
                self.base_lr * (self.gamma**time)
                for _ in self.optimizer.param_groups
            ]

    def _get_closed_form_lr(self):
        # 到了第几个stage
        steps = np.sum(self.last_epoch >= np.asarray(self.last_stage))
        return [base_lr * (self.gamma**steps) for base_lr in self.base_lrs]


class WarmupCosineAnnealingLR(LRScheduler):
    """
    Warmup Cosine Annealing Lr
    """

    def __init__(self, optimizer, max_lr, min_lr, warmup_step, total_step):
        super().__init__(optimizer)
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.warmup_step = warmup_step
        self.total_step = total_step

    def get_lr(self):
        if self.last_epoch == 0:
            return [group["lr"] for group in self.optimizer.param_groups]
        elif self.last_epoch < self.warmup_step:
            return [
                self.min_lr + (self.max_lr - self.min_lr) * self.last_epoch /
                self.warmup_step for _ in self.optimizer.param_groups
            ]
        else:
            return [
                self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (1 + np.cos(
                    (self.last_epoch - self.warmup_step) /
                    (self.total_step - self.warmup_step) * np.pi))
                for _ in self.optimizer.param_groups
            ]


class WarmupLinearLR(LRScheduler):
    """
    Warmup Linear Lr
    """

    def __init__(self, optimizer, max_lr, min_lr, warmup_step, total_step):
        super().__init__(optimizer)
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.warmup_step = warmup_step
        self.total_step = total_step

    def get_lr(self):
        if self.last_epoch == 0:
            return [group["lr"] for group in self.optimizer.param_groups]
        elif self.last_epoch < self.warmup_step:
            return [
                self.min_lr + (self.max_lr - self.min_lr) * self.last_epoch /
                self.warmup_step for _ in self.optimizer.param_groups
            ]
        else:
            return [
                self.max_lr - (self.max_lr - self.min_lr) *
                (self.last_epoch - self.warmup_step) /
                (self.total_step - self.warmup_step)
                for _ in self.optimizer.param_groups
            ]


class WarmupPowerLR(LRScheduler):

    def __init__(self,
                 optimizer,
                 max_lr,
                 min_lr,
                 warmup_step,
                 total_step,
                 power=4):
        super().__init__(optimizer)
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.warmup_step = warmup_step
        self.total_step = total_step
        self.power = power

    def get_lr(self):
        if self.last_epoch == 0:
            return [group["lr"] for group in self.optimizer.param_groups]
        elif self.last_epoch < self.warmup_step:
            return [
                self.min_lr + (self.max_lr - self.min_lr) * self.last_epoch /
                self.warmup_step for _ in self.optimizer.param_groups
            ]
        else:
            return [
                self.min_lr + (self.max_lr - self.min_lr) *
                (1 - (self.last_epoch - self.warmup_step) /
                 (self.total_step - self.warmup_step))**self.power
                for _ in self.optimizer.param_groups
            ]


def get_scheduler(optimizer, cfg):
    if cfg.SOLVER.LR_SCHEDULER == "StepScaleLR":
        return StepScaleLR(optimizer, cfg.SOLVER.MAX_LR,
                           cfg.SOLVER.LR_DECAY_STAGES,
                           cfg.SOLVER.LR_DECAY_RATE,
                           cfg.DATASET.NUM_IMAGES // cfg.DATASET.BATCH_SIZE)
    elif cfg.SOLVER.LR_SCHEDULER == "WarmupCosineAnnealingLR":
        return WarmupCosineAnnealingLR(optimizer, cfg.SOLVER.MAX_LR,
                                       cfg.SOLVER.MIN_LR, cfg.warmup_step,
                                       cfg.total_step)
    elif cfg.SOLVER.LR_SCHEDULER == "WarmupLinearLR":
        return WarmupLinearLR(optimizer, cfg.SOLVER.MAX_LR, cfg.SOLVER.MIN_LR,
                              cfg.warmup_step, cfg.total_step)
    elif cfg.SOLVER.LR_SCHEDULER == "WarmupPowerLR":
        return WarmupPowerLR(optimizer, cfg.SOLVER.MAX_LR, cfg.SOLVER.MIN_LR,
                             cfg.warmup_step, cfg.total_step)
    else:
        raise ValueError(f"Unknown scheduler: {cfg.SOLVER.LR_SCHEDULER}")
