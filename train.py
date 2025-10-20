import logging
import torch
import torch.optim as optim
from torch.nn import DataParallel

import os
import sys
from model import get_model
from dataset.dataset import get_dataloader, get_transform
from util.logging_utils import init_logger, set_map, RunTimeLogging, AverageMeter
from util.config_util import get_config, print_config
from util.lr_util import get_scheduler
from distiller import get_arcface, get_distiller, get_TLoss
from eval.verification import Verification


def train(distiller, dataloader, cfg, tbLogger, amp, t_loss=None):
    distiller = DataParallel(distiller.cuda())
    # 定义optimizer
    arcface = get_arcface(cfg)
    arcface.cuda()
    optimizer = optim.SGD(
        [{
            "params": distiller.module.get_learnable_parameters()
        }, {
            "params": arcface.parameters()
        }],
        lr=cfg.SOLVER.MAX_LR,
        momentum=cfg.SOLVER.MOMENTUM,
        weight_decay=cfg.SOLVER.WEIGHT_DECAY)

    # todo 增加随机数种子的设置，使得实验结果能够复现
    # todo 将训练的优化器和学习率更新策略保持和arcface一致，weightdecay等

    # get Lr scheduler
    cfg.total_step = cfg.DATASET.NUM_IMAGES // cfg.DATASET.BATCH_SIZE * cfg.SOLVER.EPOCHS
    if cfg.SOLVER.LR_SCHEDULER in [
            "WarmupCosineAnnealingLR", "WarmupLinearLR", "WarmupPowerLR"
    ]:
        cfg.warmup_step = cfg.DATASET.NUM_IMAGES // cfg.DATASET.BATCH_SIZE * cfg.SOLVER.WARMUP_EPOCH
    schedule_lr = get_scheduler(optimizer, cfg)

    # amp
    set_map("TRAIN")
    start_epoch = 0
    step = 0
    if cfg.RESUME.IS_RESUME:
        # 学生网络从断点继续训练
        check_point = torch.load(
            os.path.join(cfg.RESUME.RESUME_PATH,
                         f"checkpoint_{cfg.RESUME.RESUME_EPOCH}_epoch.pt"))
        start_epoch = cfg.RESUME.RESUME_EPOCH + 1
        distiller.load_state_dict(check_point['distiller_state_dict'])
        optimizer.load_state_dict(check_point['optimizer_state_dict'])
        schedule_lr.load_state_dict(check_point['scheduler_state_dict'])
        schedule_lr.last_epoch = start_epoch * len(dataloader)
        # step = cfg.DATASET.NUM_IMAGES // cfg.DATASET.BATCH_SIZE * start_epoch
        # if len(schedule_lr.stages) < len(cfg.SOLVER.LR_DECAY_STAGES):
        #     schedule_lr.last_stage = cfg.SOLVER.LR_DECAY_STAGES[len(schedule_lr.stages)]
        #     schedule_lr.stages = cfg.SOLVER.LR_DECAY_STAGES
        del check_point
    loss_am = AverageMeter()
    # 混合精度计算，降低运算成本
    run_time_logging = RunTimeLogging(frequent=cfg.SOLVER.PRINT_FREQ,
                                      total_step=cfg.total_step,
                                      batch_size=cfg.DATASET.BATCH_SIZE,
                                      start_step=step)
    ver_best_dict = {k: 0 for k in cfg.EVAL.EVAL_DATASET}
    ver = Verification(cfg)

    for epoch in range(start_epoch, cfg.SOLVER.EPOCHS):
        distiller.train()
        for _, (index, img, flip_flag, label) in enumerate(dataloader):
            optimizer.zero_grad()
            img = img.cuda()
            label = label.cuda()
            logits_t, logits_s, loss_dict = distiller(index=index,
                                                      image=img,
                                                      flip_flag=flip_flag,
                                                      epoch=epoch)
            loss = loss_dict.sum()
            arcface_loss = arcface(logits_s, label)
            loss += arcface_loss
            if t_loss is not None:
                tloss = t_loss(logits_s, logits_t).mean()
                loss += tloss
            if cfg.SOLVER.FP16:
                amp.scale(loss).backward()
                amp.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    distiller.module.get_learnable_parameters(), 5)
                amp.step(optimizer)
                amp.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    distiller.module.get_learnable_parameters(), 5)
                optimizer.step()
            loss_am.update(loss.item(), 1)
            # torch.cuda.empty_cache()
            step += 1
            schedule_lr.step()
            run_time_logging(
                step,
                arcface_loss.item(),
                loss_am,
                epoch,
                cfg.SOLVER.FP16,
                schedule_lr.get_last_lr()[0],
                amp,
            )
            loss_dict = {}
            loss_dict.update({"arcface_loss": arcface_loss.item()})
            loss_dict.update({"tloss": tloss.item() if t_loss else 0})
            loss_dict.update({
                "loss": loss_am.avg,
                'lr': schedule_lr.get_last_lr()[0]
            })
            tbLogger.log_everything(loss_dict, step=step)

        # 每隔一定的epoch在验证集上查看模型的性能
        if epoch % cfg.LOG.FREQUENCY == 0:
            set_map("EVAL")
            distiller.eval()
            ver_dict = ver.verification(distiller.module.student)
            for key, (acc, std, xnorm) in ver_dict.items():
                if acc > ver_best_dict[key]:
                    ver_best_dict[key] = acc
                logging.info(
                    f"{key}: acc {acc * 100:.2f} best acc {ver_best_dict[key] * 100:.2f}% std: {std} xnorm: {xnorm}"
                )
            set_map("TRAIN")

        if epoch % cfg.SOLVER.SAVE_STEP == 0:
            # 在测试集上计算准确度并且保存模型参数
            check_point = {
                "epoch": epoch,
                "distiller_state_dict": distiller.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": schedule_lr.state_dict()
            }
            torch.save(
                check_point,
                os.path.join(cfg.save_dir, f"checkpoint_{epoch}_epoch.pt"))
            torch.save(distiller.module.student.state_dict(),
                       os.path.join(cfg.save_dir, f"student_{epoch}_epoch.pt"))




def main(args, cfg):
    
    save_dir, tbLogger = init_logger(cfg.EXPERIMENT.LOG_DIR,
                                     cfg.EXPERIMENT.NAME)
    cfg.save_dir = save_dir
    set_map("INFO")
    logging.info(f'teacher network: {cfg.DISTILLER.TEACHER}')
    logging.info(f'student network: {cfg.DISTILLER.STUDENT}')
    teacher_model = get_model(cfg.DISTILLER.TEACHER, cfg)
    student_model = get_model(cfg.DISTILLER.STUDENT,
                              cfg,
                              blocks=(1, 4, 6, 2),
                              scale=2)
    print_config(cfg)
    teacher_model.load_state_dict(torch.load(cfg.SOLVER.TEACHER_PTH))
    # student_model.load_state_dict(torch.load("/home/power/tx/FC_distillation/models/60epoch/student_54_epoch.pt"))
    distiller = get_distiller(cfg, student_model, teacher_model)
    # dataloader = get_dataloader(rank, cfg.DATASET.DATA_DIR, cfg.DATASET.BATCH_SIZE)
    dataloader = get_dataloader(cfg.DATASET.DATA_DIR, cfg.DATASET.BATCH_SIZE)
    amp = torch.cuda.amp.GradScaler(growth_interval=100)
    # train(rank, distiller, dataloader, cfg, amp)
    t_loss = get_TLoss(cfg)

    train(distiller, dataloader, cfg, tbLogger, amp, t_loss=t_loss)
    logging.info("train finished")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='distillation')
    parser.add_argument('--config',
                        default="./config/kl_norm_TLoss_v2,res100,mv3.yaml",
                        type=str,
                        help='config file')

    args = parser.parse_args()
    cfg = get_config(args.config)
    main(args, cfg)
