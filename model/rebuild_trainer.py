import copy
import os
import time

import joblib
import torch
import yaml
from tabulate import tabulate
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from metric.rmse import RMSE
from model.rebuild_model import RebuildNet
from model.utils.trainer_utils import (
    WarmUpCosineLR,
    EarlyStopping,
    RebuildCriterion,
    CoordinateInverse,
)


class RebuildTrainer:
    """
        重构模型的训练器
        参数
           config：参数配置
    """
    def __init__(self, config):
        self.config = config
        self.task = self.config["rebuild"]["task"]
        self.sample_ratio = self.config["rebuild"]["sample_ratio"]
        self.epoch = 0
        self.Trar = int(self.config["rebuild"]["alpha"] * self.config['common']['Th'])
        self.Tsar = int(self.config["rebuild"]["beta"] * self.config['common']['Th'])
        self.device = torch.device(self.config["rebuild"]["device_str"])
        self.train_time = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime())
        self.data_dir = f"{self.config['common']['data_dir']}/rebuild"
        self.log_dir = f"{self.config['common']['log_dir']}/rebuild/{self.task['RTR']}{self.task['STR']}/{self.train_time}"
        self.checkpoint_dir = f"{self.config['common']['log_dir']}/rebuild/{self.task['RTR']}{self.task['STR']}/{self.train_time}/checkpoint"
        for path in [self.log_dir, self.checkpoint_dir]:
            if not os.path.exists(path):
                os.makedirs(path)
        self.train_data = joblib.load(f"{self.data_dir}/train_loader.pkl")
        self.val_data = joblib.load(f"{self.data_dir}/val_loader.pkl")
        self.val_meta = joblib.load(f"{self.data_dir}/val_meta.pkl")
        self.device = torch.device(self.config["Rebuild"]["device_str"])
        self.model = RebuildNet(self.config).to(self.device)
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.config["rebuild"]["default_lr"],
            weight_decay=self.config["rebuild"]["weight_decay"],
        )
        self.scheduler = WarmUpCosineLR(
            optimizer=self.optimizer,
            warmup_epoch=self.config["rebuild"]["warmup_epochs"],
            T_max=self.config["rebuild"]["epochs"] - self.config["rebuild"]["warmup_epochs"],
            lr_max=self.config["rebuild"]["default_lr"],
            lr_min=1e-8,
        ).get_lr_scheduler()
        self.criterion = RebuildCriterion(
            T=2,
            task=self.task,
            max_epoch=self.config["rebuild"]["epochs"],
            rep_grad=True,
            device=self.device,
            log_dir=self.log_dir,
            kinematic=self.config["rebuild"]["kinematic"],
            dwa=self.config["rebuild"]["dwa"]
        )
        self.early_stopping = EarlyStopping(self.config["rebuild"]["patience"], self.checkpoint_dir)
        self.lr_writer = SummaryWriter(self.log_dir)
        self.best = None

    def train(self):
        """
            训练的主函数，进行多个epoch的迭代优化，每个epoch中分别进行一次训练和一次验证
        """
        print(
            f"\n/*********************************************************************"
            f" Pre-train {self.task['RTR']}{self.task['STR']} "
            f"*********************************************************************/"
        )
        for epoch in range(self.config["rebuild"]["epochs"]):
            print(
                f"\n/------------------------------------------------- "
                f"Epoch: {epoch}/{self.config['rebuild']['epochs'] - 1}, Loss Weights: [RTR_R  RTR_K  RTR  STR] [{self.criterion.weight_buffer['RTR_R'][epoch]:.2f}  {self.criterion.weight_buffer['RTR_K'][epoch]:.2f}  {self.criterion.weight_buffer['RTR'][epoch]:.2f}  {self.criterion.weight_buffer['STR'][epoch]:.2f}]"
                f" -------------------------------------------------/"
            )
            self.epoch = epoch
            train_losses = self.train_step()
            val_losses = self.val_step()
            self.lr_writer.add_scalar("Learning Rate", self.optimizer.state_dict()["param_groups"][0]["lr"], epoch)
            self.criterion(epoch, train_losses, val_losses)
            self.early_stopping(epoch, train_losses, val_losses, self.model)
            self.scheduler.step()
            if epoch < self.config["rebuild"]["warmup_epochs"]:
                if self.early_stopping.counter == self.early_stopping.patience:
                    print(
                        f"\nStopping too early at epoch {epoch}/{self.config['rebuild']['epochs'] - 1}! Recounting..."
                    )
                    self.early_stopping.counter = 0
            elif epoch >= self.config["rebuild"]["warmup_epochs"]:
                if self.early_stopping.counter == self.early_stopping.patience:
                    print(
                        f"\n/*********************************************************************"
                        f" Earlystopping Epoch: {epoch}/{self.config['rebuild']['epochs'] - 1} "
                        f"*********************************************************************/"
                    )
                    break
                if epoch + 1 == self.config["rebuild"]["epochs"]:
                    print(
                        f"\n/*********************************************************************"
                        f" Finished Epoch: {epoch}/{self.config['rebuild']['epochs'] - 1} "
                        f"*********************************************************************/"
                    )
                    self.early_stopping.save_checkpoint()
        self.best = self.early_stopping.checkpoint[-1]
        self.config["rebuild"]["test_model_path"] = list(self.best.values())[0]
        with open(f"{self.log_dir}/config.yaml", "w", encoding="utf-8") as file:
            yaml.dump(self.config, file, default_flow_style=False, allow_unicode=True)

    def train_step(self):
        """
            执行一次训练，按批量输入所有训练集数据，进行前向和反向传播
        """
        TrainLosses = {"RTR_R": [], "RTR_K": [], "RTR": [], "STR": [], "Total": []}
        self.model.train()
        train_batch_num = len(self.train_data)
        train_bar = tqdm(range(train_batch_num))
        for i, (traj, traj_mask) in zip(train_bar, self.train_data):
            if i < int(train_batch_num * self.sample_ratio):
                traj = traj[:, :, : self.config['common']['Th'], :].to(self.device)
                traj_mask = traj_mask[:, :, : self.config['common']['Th']].to(self.device)
                traj_target = traj[:, :, : self.config['common']['Th'], :2].to(self.device)
                outputs = self.model(traj, traj_mask, traj_target)
                losses = self.criterion.compute_loss(outputs)
                self.optimizer.zero_grad()
                losses["Total"].backward()
                self.optimizer.step()
                for key in TrainLosses.keys():
                    TrainLosses[key].append(losses[key])
                lr = self.optimizer.state_dict()["param_groups"][0]["lr"]
                train_bar.set_description(
                    f"<Training>  [Learning Rate]: {lr}  [Total Loss]: {losses['Total']:.6f}  [RTR Loss]: {losses['RTR']:.6f}  [STR Loss]: {losses['STR']:.6f}"
                )
            else:
                break
        train_bar.close()
        for key in TrainLosses.keys():
            TrainLosses[key] = torch.tensor(TrainLosses[key]).mean()
        TrainLossTable = tabulate(
            [[f"{key} Train Loss" for key in TrainLosses.keys()], [f"{value:.6f}" for value in TrainLosses.values()]],
            headers="firstrow",
            tablefmt="simple_outline",
            numalign="center",
            stralign="center",
        )
        print(f"\n{TrainLossTable}\n")

        return TrainLosses

    def val_step(self):
        """
            执行一次验证，按批量输入所有验证集数据，验证模型的重构损失
        """
        ValLosses = {"RTR_R": [], "RTR_K": [], "RTR": [], "STR": [], "Total": []}
        self.model.eval()
        with torch.no_grad():
            val_batch_num = len(self.val_data)
            val_bar = tqdm(range(val_batch_num))
            for i, (traj, traj_mask), meta in zip(val_bar, self.val_data, self.val_meta):
                if i < int(val_batch_num * self.sample_ratio):
                    traj = traj[:, :, : self.config['common']['Th'], :].to(self.device)
                    traj_mask = traj_mask[:, :, : self.config['common']['Th']].to(self.device)
                    traj_target = traj[:, :, : self.config['common']['Th'], :2].to(self.device)
                    outputs = self.model(traj, traj_mask, traj_target)
                    losses = self.criterion.compute_loss(outputs)
                    for key in ValLosses.keys():
                        ValLosses[key].append(losses[key])
                    val_bar.set_description(
                        f"<Validation>  [Total Loss]: {losses['Total'].item():.6f}  [RTR Loss]: {losses['RTR'].item():.6f}  [STR Loss]: {losses['STR']:.6f}")
                else:
                    break
            val_bar.close()
            for key in ValLosses.keys():
                ValLosses[key] = torch.tensor(ValLosses[key]).mean()
            ValLossTable = tabulate(
                [[f"{key} Val Loss" for key in ValLosses.keys()], [f"{value:.6f}" for value in ValLosses.values()]],
                headers="firstrow",
                tablefmt="simple_outline",
                numalign="center",
                stralign="center",
            )
            print(f"\n{ValLossTable}\n")

            return ValLosses

    def test(self):
        """
            测试主函数，按批量输入所有验证集数据，测试预训练完成的重构模型的损失和误差
        """
        if self.config["rebuild"]["test_model_path"] is None:
            self.config["rebuild"]["test_model_path"] = list(self.best.values())[0]
            with open(f"{self.log_dir}/config.yaml", "w", encoding="utf-8") as file:
                yaml.dump(self.config, file, default_flow_style=False, allow_unicode=True)
        self.model = torch.load(self.config["rebuild"]["test_model_path"]).to(self.device)

        TestLosses = {"RTR_R": [], "RTR_K": [], "RTR": [], "STR": [], "Total": []}
        TestRMSE = {"RTR_X": RMSE().to(self.device), "RTR_Y": RMSE().to(self.device), "RTR": RMSE().to(self.device),
                    "STR_X": RMSE().to(self.device), "STR_Y": RMSE().to(self.device), "STR": RMSE().to(self.device)}
        TestTraj = {"RTR_Rebuild": [], "RTR_Kinematic": [], "RTR_TargetMask": [], "RTR_MetaIndex": [],
                    "STR_Rebuild": [], "STR_TargetMask": [], "STR_MetaIndex": []}
        self.model.eval()
        with torch.no_grad():
            val_batch_num = len(self.val_data)
            val_bar = tqdm(range(val_batch_num))
            for i, (traj, traj_mask), meta in zip(val_bar, self.val_data, self.val_meta):
                if i < int(val_batch_num * self.sample_ratio):
                    traj = traj[:, :, : self.config['common']['Th'], :].to(self.device)
                    traj_mask = traj_mask[:, :, : self.config['common']['Th']].to(self.device)
                    traj_target = traj[:, :, : self.config['common']['Th'], :2].to(self.device)
                    outputs = self.model(traj, traj_mask, traj_target)
                    losses = self.criterion.compute_loss(outputs)
                    for key in TestLosses.keys():
                        TestLosses[key].append(losses[key])
                    if outputs["RTR"]:
                        rtr_rebuild, rtr_kinematic, rtr_target, rtr_target_mask, rtr_meta_index = outputs["RTR"]
                        rtr_rebuild_m, rtr_kinematic_m, rtr_target_m = CoordinateInverse(
                            [rtr_rebuild, rtr_kinematic, rtr_target], meta,
                            rtr_meta_index).batch_inverse()
                        TestRMSE["RTR_X"].update(rtr_rebuild_m[..., 0], rtr_target_m[..., 0])
                        TestRMSE["RTR_Y"].update(rtr_rebuild_m[..., 1], rtr_target_m[..., 1])
                        TestRMSE["RTR"].update(rtr_rebuild_m, rtr_target_m)
                        TestTraj["RTR_Rebuild"].append(copy.deepcopy(rtr_rebuild_m.cpu().detach().numpy()))
                        TestTraj["RTR_Kinematic"].append(copy.deepcopy(rtr_kinematic_m.cpu().detach().numpy()))
                        TestTraj["RTR_TargetMask"].append(copy.deepcopy(rtr_target_mask.cpu().detach().numpy()))
                        TestTraj["RTR_MetaIndex"].append(copy.deepcopy(rtr_meta_index.cpu().detach().numpy()))
                    if outputs["STR"]:
                        str_rebuild, str_target, str_target_mask, str_meta_index = outputs["STR"]
                        str_rebuild_m, str_target_m = CoordinateInverse(
                            [str_rebuild, str_target], meta, str_meta_index).batch_inverse()
                        TestRMSE["STR_X"].update(str_rebuild_m[..., 0], str_target_m[..., 0])
                        TestRMSE["STR_Y"].update(str_rebuild_m[..., 1], str_target_m[..., 1])
                        TestRMSE["STR"].update(str_rebuild_m, str_target_m)
                        TestTraj["STR_Rebuild"].append(copy.deepcopy(str_rebuild_m.cpu().detach().numpy()))
                        TestTraj["STR_TargetMask"].append(copy.deepcopy(str_target_mask.cpu().detach().numpy()))
                        TestTraj["STR_MetaIndex"].append(copy.deepcopy(str_meta_index.cpu().detach().numpy()))
                    val_bar.set_description(
                        f"<Validation>  [Total Loss]: {losses['Total'].item():.6f}  [RTR Loss]: {losses['RTR'].item():.6f}  [STR Loss]: {losses['STR']:.6f}")
                else:
                    break
            val_bar.close()
            for key in TestLosses.keys():
                TestLosses[key] = torch.tensor(TestLosses[key]).mean()
            for key in TestRMSE.keys():
                TestRMSE[key] = TestRMSE[key].compute()
            TestLossTable = tabulate(
                [[key for key in TestLosses.keys()],
                 [value for value in TestLosses.values()]],
                headers="firstrow",
                tablefmt="simple_outline",
                numalign="center",
                stralign="center",
            )
            TestMetricTable = tabulate(
                [[f"{key} RMSE" for key in TestRMSE.keys()],
                 [f"{value:.6f}" for value in TestRMSE.values()]],
                headers="firstrow",
                tablefmt="simple_outline",
                numalign="center",
                stralign="center",
            )
            print(f"\nTesting model path: {self.config['rebuild']['test_model_path']}")
            print(f"\n{TestLossTable}\n")
            print(f"{TestMetricTable}\n")
            with open(f"{self.log_dir}/test_log.txt", "w") as file:
                file.write(f"\n{TestLossTable}\n")
                file.write(f"{TestMetricTable}\n")
            joblib.dump(TestTraj, f"{self.log_dir}/test_traj.pkl")
