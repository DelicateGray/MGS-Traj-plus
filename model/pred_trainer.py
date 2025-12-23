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
from model.pred_model import PredNet
from model.utils.trainer_utils import (
    WarmUpCosineLR,
    EarlyStopping,
    PredCriterion,
    CoordinateInverse,
)


class PredTrainer:
    """
        预测模型的训练器
        参数
           config：参数配置
    """
    def __init__(self, config):
        self.config = config
        self.task = self.config["pred"]["task"]
        self.pretrained_model_path = self.config["pred"]["pretrained_model_path"]
        self.sample_ratio = self.config["pred"]["sample_ratio"]
        self.epoch = 0
        self.device = torch.device(self.config["pred"]["device_str"])
        self.train_time = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime())
        self.data_dir = f"{self.config['common']['data_dir']}/pred"
        self.log_dir = f"{self.config['common']['log_dir']}/pred/{self.task['RTR']}{self.task['STR']}/{self.train_time}"
        self.checkpoint_dir = f"{self.config['common']['log_dir']}/pred/{self.task['RTR']}{self.task['STR']}/{self.train_time}/checkpoint"
        for path in [self.log_dir, self.checkpoint_dir]:
            if not os.path.exists(path):
                os.makedirs(path)
        self.train_data = joblib.load(f"{self.data_dir}/train_loader.pkl")
        self.val_data = joblib.load(f"{self.data_dir}/val_loader.pkl")
        self.test_data = joblib.load(f"{self.data_dir}/test_loader.pkl")
        self.val_meta = joblib.load(f"{self.data_dir}/val_meta.pkl")
        self.test_meta = joblib.load(f"{self.data_dir}/test_meta.pkl")
        self.model = PredNet(self.config).to(self.device)
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.config["pred"]['default_lr'],
            weight_decay=self.config["pred"]['weight_decay'],
        )
        self.scheduler = WarmUpCosineLR(
            optimizer=self.optimizer,
            warmup_epoch=self.config['pred']['warmup_epochs'],
            T_max=self.config['pred']['epochs'] - self.config['pred']['warmup_epochs'],
            lr_max=self.config['pred']['default_lr'],
            lr_min=0,
        ).get_lr_scheduler()
        self.criterion = PredCriterion(log_dir=self.log_dir)
        self.early_stopping = EarlyStopping(self.config['pred']['patience'], self.checkpoint_dir)
        self.lr_writer = SummaryWriter(self.log_dir)
        self.best = None

    def train(self):
        """
            训练的主函数，进行多个epoch的迭代优化，每个epoch中分别进行一次训练和一次验证
        """
        print(
            f"\n/*********************************************************************"
            f" Train {self.task['RTR']}{self.task['STR']} "
            f"*********************************************************************/\n"
        )
        for epoch in range(self.config['pred']['epochs']):
            print(
                f"\n/--------------------------------------------------------------------- "
                f"Epoch: {epoch}/{self.config['pred']['epochs'] - 1}"
                f" ---------------------------------------------------------------------/"
            )
            self.epoch = epoch
            TrainLosses = self.train_step()
            ValLosses = self.val_step()
            self.lr_writer.add_scalar("Learning Rate", self.optimizer.state_dict()["param_groups"][0]["lr"], epoch)
            self.criterion(epoch, TrainLosses, ValLosses)
            self.early_stopping(epoch, TrainLosses, ValLosses, self.model)
            self.scheduler.step()
            if epoch < self.config['pred']['warmup_epochs']:
                if self.early_stopping.counter == self.early_stopping.patience:
                    print(
                        f"\nStopping too early at epoch {epoch}/{self.config['pred']['warmup_epochs'] - 1}! Recounting..."
                    )
                    self.early_stopping.counter = 0
            elif epoch >= self.config['pred']['warmup_epochs']:
                if self.early_stopping.counter == self.early_stopping.patience:
                    print(
                        f"\n/*********************************************************************"
                        f" Earlystopping Epoch: {epoch}/{self.config['pred']['epochs'] - 1} "
                        f"*********************************************************************/"
                    )
                    break
                if epoch + 1 == self.config['pred']['epochs']:
                    self.early_stopping.save_checkpoint()
                    print(
                        f"\n/*********************************************************************"
                        f" Finished Epoch: {epoch}/{self.config['pred']['epochs'] - 1} "
                        f"*********************************************************************/"
                    )

        self.best = self.early_stopping.checkpoint[-1]
        self.config["pred"]["test_model_path"] = list(self.best.values())[0]
        with open(f"{self.log_dir}/config.yaml", "w", encoding="utf-8") as file:
            yaml.dump(self.config, file, default_flow_style=False, allow_unicode=True)

    def train_step(self):
        """
            执行一次训练，按批量输入所有训练集数据，进行前向和反向传播
        """
        TrainLosses = {"Total": []}
        self.model.train()
        train_batch_num = len(self.train_data)
        train_bar = tqdm(range(train_batch_num))
        for i, (traj, traj_mask) in zip(train_bar, self.train_data):
            if i < int(train_batch_num * self.sample_ratio):
                traj_his = traj[:, :, : self.config['common']['Th'], :].to(self.device)
                traj_his_mask = traj_mask[:, :, : self.config['common']['Th']].to(self.device)
                target = traj[:, 0, -self.config.Tf:, :2].to(self.device)
                pred = self.model(traj_his, traj_his_mask)
                TrainLosses["Total"].append(self.criterion.compute_loss(pred, target))
                self.optimizer.zero_grad()
                TrainLosses["Total"][-1].backward()
                self.optimizer.step()
                lr = self.optimizer.state_dict()["param_groups"][0]["lr"]

                train_bar.set_description(
                    f"<Training>  [Learning Rate]: {lr}  [Train Loss]: {TrainLosses['Total'][-1]:.6f}"
                )
            else:
                break

        train_bar.close()
        TrainLosses["Total"] = torch.tensor(TrainLosses["Total"]).mean()
        TrainLossTable = tabulate(
            [["Train Loss"], [f"{TrainLosses['Total']:.6f}"]],
            headers="firstrow",
            tablefmt="simple_outline",
            numalign="center",
            stralign="center",
        )
        print(f"\n{TrainLossTable}\n")

        return TrainLosses

    def val_step(self):
        """
            执行一次验证，按批量输入所有验证集数据，验证模型的预测损失
        """
        ValLosses = {"Total": []}
        self.model.eval()
        with torch.no_grad():
            val_batch_num = len(self.val_data)
            val_bar = tqdm(range(val_batch_num))
            for i, (traj, traj_mask), meta in zip(val_bar, self.val_data, self.val_meta):
                if i < int(val_batch_num * self.sample_ratio):
                    traj_his = traj[:, :, : self.config['common']['Th'], :].to(self.device)
                    traj_his_mask = traj_mask[:, :, : self.config['common']['Th']].to(self.device)
                    target = traj[:, 0, self.config['common']['Th']:, :2].to(self.device)
                    pred = self.model(traj_his, traj_his_mask)
                    ValLosses["Total"].append(self.criterion.compute_loss(pred, target))
                    val_bar.set_description(
                        f"<Validation>  [Val Loss]: {ValLosses['Total'][-1].item():.6f}"
                    )
                else:
                    break
            val_bar.close()
        ValLosses["Total"] = torch.tensor(ValLosses["Total"]).mean()
        ValLossTable = tabulate(
            [["Val Loss"], [f"{ValLosses['Total']:.6f}"]],
            headers="firstrow",
            tablefmt="simple_outline",
            numalign="center",
            stralign="center",
        )
        print(f"\n{ValLossTable}", )

        return ValLosses

    def test(self, model_path=None):
        """
            测试主函数，按批量输入所有测试集数据，测试训练完成的预测模型的损失和误差
        """
        if self.config["pred"]["test_model_path"] is None:
            self.config["pred"]["test_model_path"] = list(self.best.values())[0]
            with open(f"{self.log_dir}/config.yaml", "w", encoding="utf-8") as file:
                yaml.dump(self.config, file, default_flow_style=False, allow_unicode=True)
        self.model = torch.load(model_path, map_location=torch.device('cuda:0')).to(self.device)
        TestLosses = {"Total": []}
        TestMetrics = {
            "RMSE_X": [RMSE().to(self.device), RMSE().to(self.device), RMSE().to(self.device), RMSE().to(self.device),
                       RMSE().to(self.device)],
            "RMSE_Y": [RMSE().to(self.device), RMSE().to(self.device), RMSE().to(self.device), RMSE().to(self.device),
                       RMSE().to(self.device)],
            "RMSE": [RMSE().to(self.device), RMSE().to(self.device), RMSE().to(self.device), RMSE().to(self.device),
                     RMSE().to(self.device)]}
        TestTraj = {"Pred": []}
        self.model.eval()
        with torch.no_grad():
            test_bacth_num = len(self.test_data)
            test_bar = tqdm(range(test_bacth_num))
            for i, (traj, traj_mask), meta in zip(test_bar, self.test_data, self.test_meta):
                if i < int(test_bacth_num * self.sample_ratio):
                    traj_his = traj[:, :, : self.config['common']['Th'], :].to(self.device)
                    traj_his_mask = traj_mask[:, :, : self.config['common']['Th']].to(self.device)
                    target = traj[:, 0, self.config['common']['Th']:, :2].to(self.device)
                    pred = self.model(traj_his, traj_his_mask)
                    TestLosses["Total"].append(self.criterion.compute_loss(pred, target))
                    pred_m, target_m = CoordinateInverse([pred, target], meta,
                                                         torch.arange(len(meta))).batch_inverse()
                    TestTraj["Pred"].append(copy.deepcopy(pred_m.cpu().detach().numpy()))
                    for t in range(1, 6):
                        TestMetrics["RMSE_X"][t - 1].update(pred_m[..., :t * 5, 0], target_m[..., :t * 5, 0])
                        TestMetrics["RMSE_Y"][t - 1].update(pred_m[..., :t * 5, 1], target_m[..., :t * 5, 1])
                        TestMetrics["RMSE"][t - 1].update(pred_m[..., :t * 5, :], target_m[..., :t * 5, :])
                    test_bar.set_description(
                        f"<Testing>  [Test Loss]: {TestLosses['Total'][-1].item():.6f}"
                    )
                else:
                    break
            test_bar.close()
        # 合并、打印最优模型损失和指标
        TestLosses['Total'] = torch.tensor(TestLosses['Total']).mean()

        for key in TestMetrics.keys():
            for t in range(1, 6):
                TestMetrics[key][t - 1] = TestMetrics[key][t - 1].compute()
        TestLossTable = tabulate(
            [["Test Loss"], [f"{TestLosses['Total']:.6f}"]],
            headers="firstrow",
            tablefmt="simple_outline",
            numalign="center",
            stralign="center",
        )
        TestMetricTable = tabulate(
            [["Time Horizon", "RMSE_X", "RMSE_Y", "RMSE"],
             ["1s", f"{TestMetrics['RMSE_X'][0]:.6f}", f"{TestMetrics['RMSE_Y'][0]:.6f}",
              f"{TestMetrics['RMSE'][0]:.6f}"],
             ["2s", f"{TestMetrics['RMSE_X'][1]:.6f}", f"{TestMetrics['RMSE_Y'][1]:.6f}",
              f"{TestMetrics['RMSE'][1]:.6f}"],
             ["3s", f"{TestMetrics['RMSE_X'][2]:.6f}", f"{TestMetrics['RMSE_Y'][2]:.6f}",
              f"{TestMetrics['RMSE'][2]:.6f}"],
             ["4s", f"{TestMetrics['RMSE_X'][3]:.6f}", f"{TestMetrics['RMSE_Y'][3]:.6f}",
              f"{TestMetrics['RMSE'][3]:.6f}"],
             ["5s", f"{TestMetrics['RMSE_X'][4]:.6f}", f"{TestMetrics['RMSE_Y'][4]:.6f}",
              f"{TestMetrics['RMSE'][4]:.6f}"]
             ],
            headers="firstrow",
            tablefmt="simple_outline",
            numalign="center",
            stralign="center", )

        print(f"\nTesting model path: {model_path}")
        print(f"\n{TestLossTable}\n")
        print(f"{TestMetricTable}\n")
        # 保存最优模型损失和指标
        with open(f"{self.log_dir}/test_log.txt", "w") as file:
            file.write(TestLossTable)
            file.write(f"\n{TestMetricTable}")
        joblib.dump(TestTraj, f"{self.log_dir}/test_traj.pkl")
