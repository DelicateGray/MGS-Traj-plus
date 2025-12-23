# import sys
#
# sys.path.extend(["../metric"])
import copy
import os
import time

import joblib
import numpy as np
import torch
from tabulate import tabulate
from tqdm.auto import tqdm

from config import *
from metric.rmse import RMSE
from trainer_utils import CoordinateInverse


class Computer:
    def __init__(self, config, sample_ratio=1):
        self.config = config
        self.sample_ratio = sample_ratio
        self.test_time = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime())
        self.data_dir = f"{self.config['common']['data_dir']}/pred"
        self.test_data = joblib.load(f"{self.data_dir}/test_loader.pkl")
        self.test_meta = joblib.load(f"{self.data_dir}/test_meta.pkl")
        self.test_time = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime())
        self.log_dir = f"../.{self.config['common']['log_dir']}/kinematic/{self.test_time}"
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

    def test(self):
        TestTraj = {"Preds": []}
        TestMetrics = {
            "RMSE_X": [RMSE(), RMSE(), RMSE(), RMSE(), RMSE()],
            "RMSE_Y": [RMSE(), RMSE(), RMSE(), RMSE(), RMSE()],
            "RMSE": [RMSE(), RMSE(), RMSE(), RMSE(), RMSE()]}
        test_bacth_num = len(self.test_data)
        test_bar = tqdm(range(test_bacth_num))
        for i, (agent, agent_mask), meta in zip(test_bar, self.test_data, self.test_meta):
            if i < int(test_bacth_num * self.sample_ratio):
                tv_his = agent[:, 0, self.config['common']['Th'] - 1, :6]  # (B,6)
                tv_tgt = [agent[:, 0, self.config['common']['Th']: self.config['common']['Th'] + t * 5, :2] for t in
                          range(1, 6)]  # 5*[(B,5,2)]
                tv_tgt = torch.cat(tv_tgt, dim=1)  # (B,25,2)
                tv_his, tv_tgt = CoordinateInverse([tv_his, tv_tgt], meta, torch.arange(len(meta))).batch_inverse()
                tv_pred = []
                for t in range(1, 26):
                    tv_pred.append(self.kinematic_model(tv_his, t * 0.2).unsqueeze(1))  # (B,1,2)
                tv_pred = torch.cat(tv_pred, dim=1)  # (B,25,2)
                for t in range(1, 6):
                    TestMetrics["RMSE_X"][t - 1].update(tv_pred[:, :t * 5, 0], tv_tgt[:, :t * 5, 0])
                    TestMetrics["RMSE_Y"][t - 1].update(tv_pred[:, :t * 5, 1], tv_tgt[:, :t * 5, 1])
                    TestMetrics["RMSE"][t - 1].update(tv_pred[:, :t * 5, :], tv_tgt[:, :t * 5, :])
                TestTraj["Preds"].append(copy.deepcopy(tv_pred.cpu().detach().numpy()))
                test_bar.set_description(f"<Testing>")
            else:
                break
        test_bar.close()
        for i in range(5):
            TestMetrics["RMSE_X"][i] = TestMetrics["RMSE_X"][i].compute()
            TestMetrics["RMSE_Y"][i] = TestMetrics["RMSE_Y"][i].compute()
            TestMetrics["RMSE"][i] = TestMetrics["RMSE"][i].compute()
        TestTraj["Preds"] = np.concatenate(TestTraj["Preds"], axis=0)
        joblib.dump(TestTraj, f"{self.log_dir}/test_traj.pkl")
        # 合并、打印最优模型损失和指标
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
        print(f"{TestMetricTable}\n")
        # 保存最优模型损失和指标
        with open(f"{self.log_dir}/test_log.txt", "w") as file:
            file.write(f"\n{TestMetricTable}")
        return TestMetrics

    def kinematic_model(self, his_traj: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
        X0 = his_traj[..., 0]
        Y0 = his_traj[..., 1]
        Vx0 = his_traj[..., 2]
        Vy0 = his_traj[..., 3]
        Ax0 = his_traj[..., 4]
        Ay0 = his_traj[..., 5]
        X1 = X0 + Vx0 * dt + 0.5 * Ax0 * dt.square()
        Y1 = Y0 + Vy0 * dt + 0.5 * Ay0 * dt.square()
        pred = torch.cat((X1.unsqueeze(-1), Y1.unsqueeze(-1)), dim=-1)

        return pred


if __name__ == "__main__":
    config = load_config("../../config.yaml")
    k_computer = Computer(config)
    k_rmse = k_computer.test()
