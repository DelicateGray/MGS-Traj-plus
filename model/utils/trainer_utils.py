import numpy as np
import math
from LibMTL.weighting import DWA
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn


class WarmUpCosineLR:
    def __init__(self, optimizer, warmup_epoch, T_max, lr_max, lr_min):
        self.optimizer = optimizer
        self.warmup_epoch = warmup_epoch
        self.T_max = T_max
        self.lr_max = lr_max
        self.lr_min = lr_min
        self.lr_lambda = self.lr_max

    def lr_func(self, epoch):
        if epoch < self.warmup_epoch:
            self.lr_lambda = (epoch + 1) / self.warmup_epoch
        else:
            self.lr_lambda = (
                                     self.lr_min
                                     + 0.5
                                     * (self.lr_max - self.lr_min)
                                     * (1 + math.cos(math.pi * (epoch + 1 - self.warmup_epoch) / self.T_max))
                             ) / self.lr_max

        return self.lr_lambda

    def get_lr_scheduler(self):
        return LambdaLR(self.optimizer, lr_lambda=self.lr_func)


class WarmUpExponentialLR:
    def __init__(self, optimizer, warmup_epoch, base_ratio):
        self.optimizer = optimizer
        self.warmup_epoch = warmup_epoch
        self.base_ratio = base_ratio

    def lr_func(self, epoch):
        if epoch < self.warmup_epoch:
            return (epoch + 1) / self.warmup_epoch
        else:
            return self.base_ratio ** (epoch + 1 - self.warmup_epoch)

    def get_lr_scheduler(self):
        return LambdaLR(self.optimizer, lr_lambda=self.lr_func)


class EarlyStopping:
    def __init__(self, patience, checkpoint_dir):
        self.patience = patience
        self.counter = 0
        self.best_val_loss = None
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint = []
        self.epoch = 0
        self.TrainLosses = None
        self.ValLosses = None
        self.model = None

    def __call__(self, epoch, TrainLosses, ValLosses, model):
        self.epoch = epoch
        self.TrainLosses = TrainLosses
        self.ValLosses = ValLosses
        self.model = model
        if self.best_val_loss is None:
            self.save_checkpoint()
        else:
            if self.ValLosses["Total"] >= self.best_val_loss:
                self.counter += 1
                print(
                    f"Val loss update ({self.best_val_loss:.6f} --> {self.ValLosses['Total']:.6f}).EarlyStopping counter: {self.counter}/{self.patience}."
                )
            else:
                self.save_checkpoint()
                self.counter = 0

    def save_checkpoint(self):
        if self.best_val_loss is None:
            print(
                f"Val loss update (None --> {self.ValLosses['Total']:.6f}).Saving checkpoint..."
            )
        else:
            print(
                f"Val loss update ({self.best_val_loss:.6f} --> {self.ValLosses['Total']:.6f}).Saving checkpoint..."
            )
        checkpoint_path = f"{self.checkpoint_dir}/checkpoint_{self.epoch}_{self.TrainLosses['Total']:.6f}_{self.ValLosses['Total']:.6f}_.pth"
        self.best_val_loss = self.ValLosses['Total']
        self.checkpoint.append({self.epoch: checkpoint_path})
        torch.save(self.model, checkpoint_path)


class RebuildCriterion(DWA):
    def __init__(self, T, task, max_epoch, rep_grad, device, log_dir, kinematic=True, dwa=True):
        super(RebuildCriterion, self).__init__()
        self.epoch = 0
        self.T = T
        self.task = task
        self.max_epoch = max_epoch
        self.task_num = len(self.task.keys())
        self.rep_grad = rep_grad
        self.device = device
        self.loss_writer = SummaryWriter(log_dir)
        self.kinematic = kinematic
        self.dwa = dwa
        self.loss_func = nn.L1Loss()
        self.train_loss_buffer = np.zeros((self.task_num, max_epoch))
        self.weight_buffer = {"RTR_R": np.zeros(max_epoch), "RTR_K": np.zeros(max_epoch), "RTR": np.zeros(max_epoch),
                              "STR": np.zeros(max_epoch)}
        self.init_weights()

    def __call__(self, epoch, TrainLosses, ValLosses):
        # 保存当前epoch的损失
        for key1, key2 in zip(TrainLosses.keys(), ValLosses.keys()):
            if TrainLosses[key1]:
                self.loss_writer.add_scalar(f"{key1} Train Loss", TrainLosses[key1], epoch)
            if ValLosses[key2]:
                self.loss_writer.add_scalar(f"{key2} Val Loss", ValLosses[key2], epoch)
        # 保存当前epoch的权重
        if self.task["RTR"]:
            self.loss_writer.add_scalar(f"RTR_R Weight", self.weight_buffer["RTR_R"][epoch], epoch)
            self.loss_writer.add_scalar(f"RTR_K Weight", self.weight_buffer["RTR_K"][epoch], epoch)
            self.loss_writer.add_scalar(f"RTR Weight", self.weight_buffer["RTR"][epoch], epoch)
        if self.task["STR"]:
            self.loss_writer.add_scalar(f"STR Weight", self.weight_buffer["STR"][epoch], epoch)
        # 更新当前epoch的训练损失
        self.train_loss_buffer[:, epoch] = np.array([TrainLosses[key] for key in self.task.keys()])
        # 计算下一个epoch中RTR和STR任务的损失权重
        if epoch < self.max_epoch - 1:
            self.epoch = epoch + 1
            # 计算RTR任务中重构损失和运动学损失的权重
            if self.task["RTR"]:
                if self.kinematic:
                    self.weight_buffer["RTR_K"][self.epoch] = TrainLosses["RTR_R"] / (
                            TrainLosses["RTR_R"] + TrainLosses["RTR_K"])
                else:
                    self.weight_buffer["RTR_K"][self.epoch] = 0
            # 计算RTR损失和STR损失的权重
            if self.task["RTR"] and self.task["STR"]:
                if self.dwa:
                    self.weight_buffer["RTR"][self.epoch], self.weight_buffer["STR"][self.epoch] = self.backward(
                        torch.tensor([TrainLosses[key] for key in self.task.keys()], requires_grad=True).to(
                            self.device),
                        T=self.T,
                    )
            # 其余情况按照使用初始化权重，无需额外计算

    def init_weights(self):
        if self.task["RTR"] == 1:
            self.weight_buffer["RTR"] = np.ones(self.max_epoch)
            self.weight_buffer["RTR_R"] = np.ones(self.max_epoch)
            if self.kinematic:
                self.weight_buffer["RTR_K"] = np.ones(self.max_epoch)
        if self.task["STR"] == 1:
            self.weight_buffer["STR"] = np.ones(self.max_epoch)

    def compute_rtr_loss(self, rtr_out):
        rebuilding_loss, kinematic_loss, rtr_loss = torch.tensor(0.0).to(self.device), torch.tensor(0.0).to(
            self.device), torch.tensor(0.0).to(self.device)
        if rtr_out:
            rtr_rebuild, rtr_kinematic, rtr_target, _, _ = rtr_out
            rebuilding_loss = self.loss_func(rtr_rebuild, rtr_target)
            kinematic_loss = self.loss_func(rtr_kinematic, rtr_target)
            rtr_loss = self.weight_buffer["RTR_R"][self.epoch] * rebuilding_loss + self.weight_buffer["RTR_K"][
                self.epoch] * kinematic_loss

        return rebuilding_loss, kinematic_loss, rtr_loss

    def compute_str_loss(self, str_out):
        str_loss = torch.tensor(0.0).to(self.device)
        if str_out:
            str_rebuild, str_target, _, _ = str_out
            str_loss = self.loss_func(str_rebuild, str_target)
        return str_loss

    def compute_loss(self, outputs):
        losses = {"RTR_R": 0, "RTR_K": 0, "RTR": 0, "STR": 0, "Total": 0}
        losses["RTR_R"], losses["RTR_K"], losses["RTR"] = self.compute_rtr_loss(outputs["RTR"])
        losses["STR"] = self.compute_str_loss(outputs["STR"])
        losses["Total"] = self.weight_buffer["RTR"][self.epoch] * losses["RTR"] + self.weight_buffer["STR"][
            self.epoch] * losses["STR"]

        return losses


class PredCriterion:
    def __init__(self, log_dir):
        super(PredCriterion, self).__init__()
        self.loss_writer = SummaryWriter(log_dir)
        self.loss_func = nn.L1Loss()

    def __call__(self, epoch, TrainLoss, ValLoss):
        self.loss_writer.add_scalar(f"Train Loss", TrainLoss["Total"], epoch)
        self.loss_writer.add_scalar(f"Val Loss", ValLoss["Total"], epoch)

    def compute_loss(self, pred, target):
        loss = self.loss_func(pred, target)

        return loss


class CoordinateInverse:
    def __init__(self, seqs, meta, meta_index):
        self.seqs = seqs
        self.meta = meta
        self.meta_index = meta_index

    def batch_inverse(self):
        for i in range(self.meta_index.shape[0]):
            for seq in self.seqs:
                seq[i, ...] = self.scenario_inverse(seq[i, ...], self.meta[self.meta_index[i]])
        return self.seqs

    def scenario_inverse(self, seq, meta):
        seq_inv = seq
        xc, yc = meta["centerPoint"]
        xmin, xmax, ymin, ymax = meta["axisRange"]

        seq_inv[..., 0] = seq_inv[..., 0] * (xmax - xmin) + xc
        seq_inv[..., 1] = seq_inv[..., 1] * (ymax - ymin) + yc
        if seq_inv.shape[-1] > 2:
            for idx in range(2, seq_inv.shape[-1]):
                if idx == 2 or idx == 4:
                    seq_inv[..., idx] *= (xmax - xmin)
                elif idx == 3 or idx == 5:
                    seq_inv[..., idx] *= (ymax - ymin)

        return seq_inv
