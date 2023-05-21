import math
import sys
import time

import torch

import train_utils.distributed_utils as utils
from .coco_eval import EvalCOCOMetric
import numpy as np


def train_one_epoch(model, optimizer, data_loader, device, epoch,
                    print_freq=5, warmup=False, scaler=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    lr_scheduler = None
    if epoch == 0 and warmup is True:  # 当训练第一轮（epoch=0）时，启用warmup训练方式，可理解为热身训练
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = utils.warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

    mloss = torch.zeros(1).to(device)  # mean losses
    loss_all = []
    for i, [images, targets] in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        task_loss = []
        # 混合精度训练上下文管理器，如果在CPU环境中不起任何作用
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            loss_dict = model(images, targets)

            # losses = sum(loss for loss in loss_dict.values())
            task_loss = [loss_dict['loss_classifier'],
                         loss_dict['loss_box_reg'],
                         loss_dict['loss_mask'],
                         loss_dict['loss_objectness'],
                         loss_dict['loss_rpn_box_reg'],
                         loss_dict['loss_biomass']]
            task_loss = torch.stack(task_loss)

            weighted_task_loss = torch.sum(task_loss[0:5])+torch.mul(model.weights, task_loss[5])
            # print(model.weights)
            losses = torch.sum(weighted_task_loss)
            # loss_value_all = []

            if epoch == 0:
                # set L(0)
                if torch.cuda.is_available():
                    initial_task_loss = task_loss.data.cpu().detach().numpy(),
                else:
                    initial_task_loss = task_loss.data

        # reduce losses over all GPUs for logging purpose
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        loss_value_all = [loss_dict_reduced['loss_classifier'].cpu().detach().numpy(),
                          loss_dict_reduced['loss_box_reg'].cpu().detach().numpy(),
                          loss_dict_reduced['loss_mask'].cpu().detach().numpy(),
                          loss_dict_reduced['loss_biomass'].cpu().detach().numpy(),
                          loss_dict_reduced['loss_objectness'].cpu().detach().numpy(),
                          loss_dict_reduced['loss_rpn_box_reg'].cpu().detach().numpy()]
        if i == 0:
            loss_all = np.array(loss_value_all)
        else:
            loss_all = (loss_all * i + np.array(loss_value_all))/(i+1)

        loss_value = losses_reduced.item()
        # 记录训练损失
        mloss = (mloss * i + loss_value) / (i + 1)  # update mean losses

        if not math.isfinite(loss_value):  # 当计算的损失为无穷大时停止训练
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            losses.backward()
            optimizer.step()

        if lr_scheduler is not None:  # 第一轮使用warmup训练方式
            lr_scheduler.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        now_lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=now_lr)

    return mloss, now_lr, loss_value_all


@torch.no_grad()
def evaluate(model, data_loader, device):
    cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test: "

    det_metric = EvalCOCOMetric(data_loader.dataset.coco, iou_type="bbox", results_file_name="det_results.json")
    seg_metric = EvalCOCOMetric(data_loader.dataset.coco, iou_type="segm", results_file_name="seg_results.json")
    for image, targets in metric_logger.log_every(data_loader, 100, header):
        image = list(img.to(device) for img in image)

        # 当使用CPU时，跳过GPU相关指令
        if device != torch.device("cpu"):
            torch.cuda.synchronize(device)

        model_time = time.time()
        outputs = model(image)

        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        model_time = time.time() - model_time

        det_metric.update(targets, outputs)
        seg_metric.update(targets, outputs)
        metric_logger.update(model_time=model_time)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    # 同步所有进程中的数据
    det_metric.synchronize_results()
    seg_metric.synchronize_results()

    if utils.is_main_process():
        coco_info = det_metric.evaluate()
        seg_info = seg_metric.evaluate()
    else:
        coco_info = None
        seg_info = None

    return coco_info, seg_info
