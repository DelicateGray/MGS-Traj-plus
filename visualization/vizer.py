import sys

sys.path.extend(["../highd/api", "../model/utils"])
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import seaborn as sns
import pandas as pd
import numpy as np
import torch
from tqdm.auto import tqdm
import joblib
from tensorboard.backend.event_processing import event_accumulator
from typing import List
from highd.api import main, read_csv
from viz_utils import plot_line, plot_point, plot_bbox, plot_triangle, plot_fillspace, show_background_image, get_sa_id, \
    get_scenario_data, get_data_from_log
from model.utils.trainer_utils import CoordinateInverse
from config import *

matplotlib.rcParams["font.family"] = 'Microsoft YaHei'


class PredTrajVizer:
    """
        预测轨迹可视化
        参数
            traj_path: 轨迹数据路径
            meta_path: 轨迹元数据路径
            save_dir: 保存路径，默认为None 
    """

    def __init__(self, traj_path: str, meta_path: str, save_dir: str = None) -> None:
        self.config = load_config("../config.yaml")
        self.highd_dir = f"{self.config['common']['data_dir']}/data"
        self.save_dir = save_dir
        self.highd_args = main.create_args()
        self.pred = np.concatenate(joblib.load(traj_path)["Pred"], axis=0)
        self.scenario_num = self.pred.shape[0]
        self.meta = joblib.load(meta_path)
        self.bbox_style = {"sv": dict(facecolor="#D79E58", fill=True, edgecolor="k", zorder=19),
                           "tv": dict(facecolor="#82B366", fill=True, edgecolor="k", zorder=19)}
        self.triangle_style = dict(facecolor="k", fill=True, edgecolor="k", lw=0.1, alpha=0.6,
                                   zorder=19)
        self.traj_style = {"gt": dict(color="g", lw=2, ls='-', zorder=10),
                           "pred": dict(color="r", lw=2, ls="--", zorder=10)}
        self.marker_style = {"gt": dict(c="g", s=45, marker='o', edgecolor='k', zorder=11),
                             "pred": dict(c="r", s=45, marker='x', zorder=11)}

    def plot_one_scenario(self, scenario_id: int) -> None:
        """
            绘制一个轨迹预测场景
            参数
                scenario_id: 轨迹预测场景id 
        """
        # 初始化画布及子图
        fig, ax = plt.subplots(1, 1)
        fig.set_size_inches(32, 4)
        # 读取场景数据
        meta = self.meta[(scenario_id + 1) // self.config['common']['B']][
            (scenario_id + 1) % self.config['common']['B'] - 1]
        record_id = str(meta["recordID"]).zfill(2)
        self.highd_args["input_path"] = f"{self.highd_dir}/{record_id}_tracks.csv"
        self.highd_args["background_image"] = f"{self.highd_dir}/{record_id}_highway.png"
        tracks = read_csv.read_track_csv(self.highd_args)
        last_his_frame = meta["trackFrame"][0][self.config['common']['Th'] * 5 - 1]
        tv_id = int(meta["trackID"][0])
        sv_id = get_sa_id(tracks[tv_id - 1], last_his_frame)
        # if len(meta["trackID"]) > 1:
        #     sv_id = np.intersect1d(sv_id, meta["trackID"][1:])
        vehicle_id = np.insert(tv_id, 1, sv_id)
        scenario = {"bbox": [], "triangle": [], "traj": [], "marker": []}
        # 处理场景数据
        for i, vid in enumerate(vehicle_id):
            track = tracks[int(vid - 1)]
            # 缩放坐标以适应背景图像尺寸
            track["bbox"] /= (0.10106 * 4)
            bbox = track["bbox"][np.isin(track["frame"], last_his_frame)][0]
            vx = track["xVelocity"][np.isin(track["frame"], last_his_frame)][0]
            x, y, w, h = bbox
            triangle_x, triangle_y = [], [y, y + h, y + h / 2]
            if vx < 0:
                x -= w / 2
                triangle_x = [x + w / 5, x + w / 5, x]
            else:
                x += w / 2
                triangle_x = [x + w * 4 / 5, x + w * 4 / 5, x + w]

            triangle = np.array([triangle_x, triangle_y]).T
            bbox[0] = x
            scenario["bbox"].append(bbox)
            scenario["triangle"].append(triangle)
            if i == 0:
                ta_frame = meta["trackFrame"][i]
                Xleftup, Yleftup, Width, Height = track["bbox"][np.isin(track["frame"], ta_frame)].T
                tv_gt = np.array([Xleftup + Width / 2, Yleftup + Height / 2]).T
                tv_pred = self.pred[scenario_id, ...] / (0.10106 * 4)
                tv_gt_marker = [tv_gt[i * 25, :] for i in range(8)] + [tv_gt[-1, :]]
                tv_pred_marker = [tv_pred[i * 5, :] for i in range(5)] + [tv_pred[-1, :]]
                scenario["traj"].append(tv_gt)
                scenario["traj"].append(tv_pred)
                scenario["marker"].append(np.array(tv_gt_marker))
                scenario["marker"].append(np.array(tv_pred_marker))
        # 绘制场景可视化图像
        show_background_image(ax, self.highd_args["background_image"])
        for j, (bbox, triangle) in enumerate(zip(scenario["bbox"], scenario["triangle"])):
            if j == 0:
                scenario["bbox"][j] = plot_bbox(bbox, self.bbox_style["tv"])
                scenario["triangle"][j] = plot_triangle(triangle, self.triangle_style)
                (scenario["traj"][0],) = plot_line(ax, scenario["traj"][0], self.traj_style["gt"])
                (scenario["traj"][1],) = plot_line(ax, scenario["traj"][1], self.traj_style["pred"])
                scenario["marker"][0] = plot_point(ax, scenario["marker"][0], self.marker_style["gt"])
                scenario["marker"][1] = plot_point(ax, scenario["marker"][1], self.marker_style["pred"])
            else:
                scenario["bbox"][j] = plot_bbox(bbox, self.bbox_style["sv"])
                scenario["triangle"][j] = plot_triangle(triangle, self.triangle_style)
            ax.add_patch(scenario["bbox"][j])
            ax.add_patch(scenario["triangle"][j])
            # 绘制公共区域
        ax.set_title(f"Pred-{scenario_id}", color="k", size=20)
        ax.xaxis.set_tick_params(length=2, color="k", labelcolor="k", labelsize=20)
        ax.yaxis.set_tick_params(length=2, color="k", labelcolor="k", labelsize=20)
        plt.subplots_adjust(wspace=0.5, hspace=0.5)  # 增加宽度和高度间隔
        plt.tight_layout()  # 确保紧凑布局

    def plot_scenarios(self, scenario_ids: np.ndarray[int] = np.array([])):
        """
            绘制多个轨迹预测场景
            参数
                scenario_ids:轨迹预测场景id数组
        """
        if scenario_ids.shape[0] == 0:
            scenario_ids = np.random.randint(0, self.scenario_num, 10)
        print(f"Pred scenario id: {scenario_ids}")
        for index in scenario_ids:
            self.plot_one_scenario(index)
            if self.save_dir:
                plt.savefig(f"{self.save_dir}/pred_scenario_{index}.png", dpi=300, bbox_inches="tight",
                            transparent=True)
            plt.show()


class RebuildTrajVizer:
    """
        重构轨迹可视化
        参数
            traj_path: 轨迹数据路径
            meta_path: 轨迹元数据路径
            save_dir: 保存路径，默认为None
    """

    def __init__(self, traj_path: str, meta_path: str, save_dir: str = None) -> None:
        self.config = load_config("../config.yaml")
        self.highd_dir = f"{self.config['common']['data_dir']}/data"
        self.save_dir = save_dir
        self.highd_args = main.create_args()
        self.traj = joblib.load(traj_path)
        self.meta = joblib.load(meta_path)
        self.scenario_num = len(self.meta) * self.config['common']['B']

        self.bbox_style = {"tv": dict(facecolor="#82B366", fill=True, edgecolor="k", zorder=19),
                           "sv": dict(facecolor="#D79E58", fill=True, edgecolor="k", zorder=19)}
        self.triangle_style = dict(facecolor="k", fill=True, edgecolor="k", lw=0.1, alpha=0.6,
                                   zorder=19)
        self.traj_style = {"tv": dict(color="green", lw=2, ls='--', zorder=10),
                           "sv": dict(color="orange", lw=2, ls='--', zorder=10)}
        self.rebuild_marker_style = {"tv": dict(c="red", s=45, marker='x', zorder=11),
                                     "sv": dict(c="red", s=45, marker='x', zorder=11)}
        self.unmask_marker_style = {"tv": dict(c="green", s=45, marker='o', edgecolor='black', zorder=11),
                                    "sv": dict(c="orange", s=45, marker='o', edgecolor='black', zorder=11)}

    def plot_one_scenario(self, scenario_id: int, task: str = "RTR") -> None:
        """
            绘制一个轨迹重构场景
            参数
                scenario_id: 轨迹重构场景id 
        """
        # 初始化画布及子图
        fig, ax = plt.subplots(1, 1, figsize=(32, 4), dpi=300)
        # 读取场景数据
        data = get_scenario_data(self.traj, self.meta, scenario_id, task, self.config['common']['B'])
        scenario_rebuild = data["Rebuild"]
        scenario_his_mask = data["HisMask"]
        scenario_meta = data["Meta"]

        record_id = str(scenario_meta["recordID"]).zfill(2)
        self.highd_args["input_path"] = f"{self.highd_dir}/{record_id}_tracks.csv"
        self.highd_args["background_image"] = f"{self.highd_dir}/{record_id}_highway.png"
        record_track = read_csv.read_track_csv(self.highd_args)
        scenario_element = {"bbox": [], "triangle": [], "his": [], "his_rebuild": [], "his_unmask": []}
        # 处理场景数据
        for i, vid in enumerate(scenario_meta["trackID"]):
            vehicle_track = record_track[int(vid - 1)]
            # 缩放坐标以适应背景图像尺寸
            vehicle_track["bbox"] /= (0.10106 * 4)
            vehicle_his_frame = scenario_meta["trackFrame"][i][:self.config['common']['Th'] * 5]
            Xleftup, Yleftup, Width, Height = vehicle_track["bbox"][
                np.isin(vehicle_track["frame"], vehicle_his_frame)].T
            vehicle_his = []
            for j in range(len(Xleftup)):
                if j % 5 == 0:
                    vehicle_his.append([Xleftup[j] + Width[j] / 2, Yleftup[j] + Height[j] / 2])
            vehicle_his = np.array(vehicle_his)
            vehicle_his_mask = scenario_his_mask[i, ...]
            vehicle_rebuild = scenario_rebuild[i, ...] / (0.10106 * 4)
            vehicle_unmask = vehicle_his[vehicle_his_mask]
            vehicle_his[~vehicle_his_mask] = vehicle_rebuild
            vehicle_bbox = vehicle_track["bbox"][np.isin(vehicle_track["frame"], vehicle_his_frame[-1])][0]
            vx = vehicle_track["xVelocity"][np.isin(vehicle_track["frame"], vehicle_his_frame[-1])][0]
            x, y, w, h = vehicle_bbox
            triangle_x, triangle_y = [], []
            if vx < 0:
                x = vehicle_his[-1, 0] - w
                y = vehicle_his[-1, 1] + h / 2
                triangle_x = [x + w / 5, x + w / 5, x]
            else:
                x = vehicle_his[-1, 0]
                y = vehicle_his[-1, 1] + h / 2
                triangle_x = [x + w * 4 / 5, x + w * 4 / 5, x + w]
            triangle_y = [y, y - h, y - h / 2]
            vehicle_triangle = np.array([triangle_x, triangle_y]).T
            vehicle_bbox[0] = x
            scenario_element["bbox"].append(vehicle_bbox)
            scenario_element["triangle"].append(vehicle_triangle)
            scenario_element["his"].append(vehicle_his)
            scenario_element["his_rebuild"].append(vehicle_rebuild)
            scenario_element["his_unmask"].append(vehicle_unmask)
        # 绘制场景可视化图像
        show_background_image(ax, self.highd_args["background_image"])
        for j, (vehicle_bbox, vehicle_triangle) in enumerate(
                zip(scenario_element["bbox"], scenario_element["triangle"])):
            if j == 0:
                scenario_element["bbox"][j] = plot_bbox(vehicle_bbox, self.bbox_style["tv"])

                scenario_element["his_rebuild"][j] = plot_point(ax, scenario_element["his_rebuild"][j],
                                                                self.rebuild_marker_style["tv"])
                scenario_element["his_unmask"][j] = plot_point(ax, scenario_element["his_unmask"][j],
                                                               self.unmask_marker_style["tv"])
                (scenario_element["his"][j],) = plot_line(ax, scenario_element["his"][j], self.traj_style["tv"])
            else:
                scenario_element["bbox"][j] = plot_bbox(vehicle_bbox, self.bbox_style["sv"])
                scenario_element["his_rebuild"][j] = plot_point(ax, scenario_element["his_rebuild"][j],
                                                                self.rebuild_marker_style["sv"])
                scenario_element["his_unmask"][j] = plot_point(ax, scenario_element["his_unmask"][j],
                                                               self.unmask_marker_style["sv"])
                (scenario_element["his"][j],) = plot_line(ax, scenario_element["his"][j], self.traj_style["sv"])

            scenario_element["triangle"][j] = plot_triangle(vehicle_triangle, self.triangle_style)
            ax.add_patch(scenario_element["bbox"][j])
            ax.add_patch(scenario_element["triangle"][j])
        # 绘制公共区域
        ax.set_title(f"{task} Scenario {scenario_id}", color="k", size=20)
        ax.xaxis.set_tick_params(length=2, color="k", labelcolor="k", labelsize=20)
        ax.yaxis.set_tick_params(length=2, color="k", labelcolor="k", labelsize=20)
        plt.subplots_adjust(wspace=0.5, hspace=0.5)  # 增加宽度和高度间隔
        plt.tight_layout()  # 确保紧凑布局

    def plot_scenarios(self, scenario_ids: np.ndarray[int] = np.array([]), task="RTR"):
        """
            绘制多个轨迹重构场景
            参数
                scenario_ids: 轨迹重构场景id数组 
        """
        if scenario_ids.shape[0] == 0:
            scenario_ids = np.random.randint(0, self.scenario_num, 10)
        print(f"{task} scenario id: {scenario_ids}")
        for index in scenario_ids:
            self.plot_one_scenario(index, task)
            if self.save_dir:
                plt.savefig(f"{self.save_dir}/{task}_scenario_{index}.png", dpi=300, bbox_inches="tight",
                            transparent=True)
            plt.show()


class RebuildLogVizer:
    """
        重构预训练参数可视化
        参数
            save_dir: 保存路径
    """

    def __init__(self, save_dir: str = None):
        self.save_dir = save_dir

    def plot_rebuild_ax1_two(self, weight_logs: List[str], titles: List[str] = None):
        """
        只画两个“ax[1]”子图：每个子图包含 RTR/STR 权重随 epoch 的曲线
        参数
            weight_logs: 长度为2的事件文件列表（都像 ...13832.0 这种）
            titles: 可选，两个子图的标题
        """
        assert len(weight_logs) == 2, "weight_logs 需要两个事件文件路径"
        fig, ax = plt.subplots(1, 2, figsize=(10, 4), dpi=300)

        for i, log_path in enumerate(weight_logs):
            log = event_accumulator.EventAccumulator(log_path)
            log.Reload()
            # 只取 ax[1] 里用到的两个量
            rtr_epoch, rtr_weight = get_data_from_log(log.Scalars("RTR Weight"))
            str_epoch, str_weight = get_data_from_log(log.Scalars("STR Weight"))

            line_rtr = {"x": rtr_epoch, "y": rtr_weight, "color": "cornflowerblue"}
            line_str = {"x": str_epoch, "y": str_weight, "color": "indianred"}

            ax[i].plot(line_rtr["x"], line_rtr["y"], c=line_rtr["color"], lw=1.5, marker="o", mfc="white", ms=5,
                       label="RTR权重")
            ax[i].plot(line_str["x"], line_str["y"], c=line_str["color"], lw=1.5, marker="o", mfc="white", ms=5,
                       label="STR权重")
            plot_fillspace(ax[i], [line_rtr, line_str])  # 复用你已有的阴影工具

            ax[i].legend(loc='upper right', bbox_to_anchor=(0.98, 0.98), fontsize=15, labelcolor="black", frameon=False)
            ax[i].set_xlabel("训练周期", color="black", size=20)
            ax[i].set_ylabel("权重", color="black", size=20)

            # 网格 & 刻度风格与原工程保持一致
            ax[i].grid(ls="--", lw=0.5, color="black")
            ax[i].xaxis.set_tick_params(length=2, color="black", labelcolor="black", labelsize=15)
            ax[i].yaxis.set_tick_params(length=2, color="black", labelcolor="black", labelsize=15)

            # 在 x 轴标签正下方标注 (a)/(b)
            ax[i].text(0.5, -0.25, "(a)" if i == 0 else "(b)", transform=ax[i].transAxes,
                       ha="center", va="top", fontsize=14)

            # 可选标题
            if titles and len(titles) == 2:
                ax[i].set_title(titles[i], color="black", size=18)

        plt.subplots_adjust(wspace=0.5, hspace=0.5)
        plt.tight_layout()
        if self.save_dir:
            plt.savefig(f"{self.save_dir}/rebuild_ax1_two.png", dpi=300, bbox_inches="tight", transparent=True)
        plt.show()

    def plot_rebuild_loss(self, log_pathes: List[str]):
        """
            重构损失可视化
            参数
                log_pathes: 重构损失日志路径列表
        """
        fig, ax = plt.subplots(1, 3, figsize=(15, 4), dpi=300)
        log = event_accumulator.EventAccumulator(log_pathes[0])
        log.Reload()
        rtr_r_epoch, rtr_r_loss = get_data_from_log(log.Scalars("RTR_R Val Loss"))
        rtr_k_epoch, rtr_k_loss = get_data_from_log(log.Scalars("RTR_K Val Loss"))
        rtr_epoch, rtr_loss = get_data_from_log(log.Scalars("RTR Val Loss"))
        str_epoch, str_loss = get_data_from_log(log.Scalars("STR Val Loss"))
        total_epoch, total_loss = get_data_from_log(log.Scalars("Total Val Loss"))
        line1 = {"x": rtr_r_epoch, "y": rtr_r_loss, "color": "cornflowerblue", "label": "重构损失"}
        line2 = {"x": rtr_k_epoch, "y": rtr_k_loss, "color": "indianred", "label": "运动学损失"}
        line3 = {"x": rtr_epoch, "y": rtr_loss, "color": "cornflowerblue", "label": "RTR损失"}
        line4 = {"x": str_epoch, "y": str_loss, "color": "indianred", "label": "STR损失"}
        line5 = {"x": total_epoch, "y": total_loss, "color": "indianred", "label": "总损失"}
        # 绘制RTR任务损失
        ax[0].plot(line1["x"], line1["y"], c=line1["color"], lw=1.5, marker="o", mfc="white", ms=5,
                   label=line1["label"])
        ax[0].plot(line2["x"], line2["y"], c=line2["color"], lw=1.5, marker="o", mfc="white", ms=5,
                   label=line2["label"])
        plot_fillspace(ax[0], [line1, line2])
        ax[0].legend(loc='upper right', bbox_to_anchor=(0.98, 0.98), fontsize=15, labelcolor="black",
                     frameon=False)
        ax[0].set_xlabel("训练周期", color="black", size=20)
        ax[0].set_ylabel("损失", color="black", size=20)
        # ax[0].set_title("RTR损失", color="black", size=20)
        # 绘制RTR和STR任务损失
        ax[1].plot(line3["x"], line3["y"], c=line3["color"], lw=1.5, marker="o", mfc="white", ms=5,
                   label=line3["label"])
        ax[1].plot(line4["x"], line4["y"], c=line4["color"], lw=1.5, marker="o", mfc="white", ms=5,
                   label=line4["label"])
        plot_fillspace(ax[1], [line3, line4])
        ax[1].legend(loc='upper right', bbox_to_anchor=(0.98, 0.98), fontsize=15, labelcolor="black",
                     frameon=False)
        ax[1].set_xlabel("训练周期", color="black", size=20)
        ax[1].set_ylabel("损失", color="black", size=20)
        # ax[1].set_title("RTR和STR损失", color="black", size=20)
        # 绘制总损失
        ax[2].plot(line5["x"], line5["y"], c=line5["color"], lw=1.5, marker="o", mfc="white", ms=5,
                   label=line5["label"])
        plot_fillspace(ax[2], [line5, {"x": total_epoch, "y": np.zeros_like(line5["y"]), "color": "black"}])
        ax[2].legend(loc='upper right', bbox_to_anchor=(0.98, 0.98), fontsize=15, labelcolor="black",
                     frameon=False)
        ax[2].set_xlabel("训练周期", color="black", size=20)
        ax[2].set_ylabel("损失", color="black", size=20)
        # ax[2].set_title("轨迹重构总损失", color="black", size=20)
        # 所有子图参数设置
        for i in range(ax.shape[0]):
            ax[i].grid(ls="--", lw=0.5, color="black")
            ax[i].xaxis.set_tick_params(length=2, color="black", labelcolor="black", labelsize=15)
            ax[i].yaxis.set_tick_params(length=2, color="black", labelcolor="black", labelsize=15)
        plt.subplots_adjust(wspace=0.5, hspace=0.5)  # 增加宽度和高度间隔
        plt.tight_layout()  # 确保紧凑布局
        if self.save_dir:
            plt.savefig(f"{self.save_dir}/rebuild_loss.png", dpi=300, bbox_inches="tight", transparent=True)
        plt.show()

    def plot_rebuild_weight_lr(self, log_pathes: List[str]):
        """
            重构损失权重和学习率可视化
            参数
                log_pathes: 重构损失日志路径列表
        """
        log1 = event_accumulator.EventAccumulator(log_pathes[0])
        log2 = event_accumulator.EventAccumulator(log_pathes[1])
        log1.Reload()
        log2.Reload()
        lr_epoch, lr = get_data_from_log(log2.Scalars("Learning Rate"))
        if len(lr_epoch) > 50:
            lr_epoch = lr_epoch[:-1]
            lr = lr[:-1] * 5e-5
        rtr_r_epoch, rtr_r_weight = get_data_from_log(log1.Scalars("RTR_R Weight"))
        rtr_k_epoch, rtr_k_weight = get_data_from_log(log1.Scalars("RTR_K Weight"))
        rtr_epoch, rtr_weight = get_data_from_log(log1.Scalars("RTR Weight"))
        str_epoch, str_weight = get_data_from_log(log1.Scalars("STR Weight"))
        line1 = {"x": rtr_r_epoch, "y": rtr_r_weight, "color": "cornflowerblue"}
        line2 = {"x": rtr_k_epoch, "y": rtr_k_weight, "color": "indianred"}
        line3 = {"x": rtr_epoch, "y": rtr_weight, "color": "cornflowerblue"}
        line4 = {"x": str_epoch, "y": str_weight, "color": "indianred"}
        line5 = {"x": lr_epoch, "y": lr, "color": "indianred"}
        # 初始化图布及子图
        fig, ax = plt.subplots(1, 3, figsize=(15, 4), dpi=300)
        # 绘制RTR任务权重图
        ax[0].plot(rtr_r_epoch, rtr_r_weight, c=line1["color"], lw=1.5, marker="o", mfc="white", ms=5,
                   label="重构权重")
        ax[0].plot(rtr_k_epoch, rtr_k_weight, c=line2["color"], lw=1.5, marker="o", mfc="white", ms=5,
                   label="运动学权重")
        plot_fillspace(ax[0], [line1, line2])
        ax[0].legend(loc='upper right', bbox_to_anchor=(0.98, 0.98), fontsize=15, labelcolor="black",
                     frameon=False)
        ax[0].set_xlabel("训练周期", color="black", size=20)
        ax[0].set_ylabel("权重", color="black", size=20)
        # ax[0].set_title("RTR权重", color="black", size=20)
        # 绘制RTR和STR任务权重图
        ax[1].plot(rtr_epoch, rtr_weight, c=line3["color"], lw=1.5, marker="o", mfc="white", ms=5,
                   label="RTR权重")
        ax[1].plot(str_epoch, str_weight, c=line4["color"], lw=1.5, marker="o", mfc="white", ms=5,
                   label="STR权重")
        plot_fillspace(ax[1], [line3, line4])
        ax[1].legend(loc='upper right', bbox_to_anchor=(0.98, 0.98), fontsize=15, labelcolor="black",
                     frameon=False)
        ax[1].set_xlabel("训练周期", color="black", size=20)
        ax[1].set_ylabel("权重", color="black", size=20)
        # ax[1].set_title("RTR和STR权重", color="black", size=20)
        # 绘制重构任务学习率
        ax[2].plot(line5["x"], line5["y"], c=line5["color"], lw=1.5, marker="o", mfc="white", ms=5,
                   label="学习率")
        plot_fillspace(ax[2], [line5, {"x": lr_epoch, "y": np.zeros_like(lr_epoch), "color": "black"}])
        ax[2].legend(loc='upper right', bbox_to_anchor=(0.98, 0.98), fontsize=15, labelcolor="black",
                     frameon=False)
        ax[2].set_xlabel("训练周期", color="black", size=20)
        ax[2].set_ylabel("学习率", color="black", size=20)
        # ax[2].set_title("轨迹重构学习率", color="black", size=20)
        # 所有子图参数设置
        for i in range(ax.shape[0]):
            ax[i].grid(ls="--", lw=0.5, color="black")
            ax[i].xaxis.set_tick_params(length=2, color="black", labelcolor="black", labelsize=15)
            ax[i].yaxis.set_tick_params(length=2, color="black", labelcolor="black", labelsize=15)
            # ax[i].set_xlabel("Epoch", color="black", size=20)
            # ax[i].set_ylabel("Value", color="black", size=20)
        plt.subplots_adjust(wspace=0.5, hspace=0.5)  # 增加宽度和高度间隔
        plt.tight_layout()  # 确保紧凑布局
        if self.save_dir:
            plt.savefig(f"{self.save_dir}/rebuild_weight_lr.png", dpi=300, bbox_inches="tight", transparent=True)
        plt.show()


class PredLogVizer:
    """
        预测损失权重和学习率可视化
        参数
            log_pathes: 预测损失及学习率日志路径列表
    """

    def __init__(self, log_pathes: List[str], **kwards):
        self.log_pathes = log_pathes
        self.kwards = kwards

    def plot_pred_loss_lr(self):
        log1 = event_accumulator.EventAccumulator(self.log_pathes[0])
        log2 = event_accumulator.EventAccumulator(self.log_pathes[1])
        log1.Reload()
        log2.Reload()
        loss_epoch, loss = get_data_from_log(log1.Scalars("Val Loss"))
        lr_epoch, lr = get_data_from_log(log2.Scalars("Learning Rate"))
        if len(lr_epoch) > 50:
            lr_epoch = lr_epoch[:-1]
            lr = lr[:-1] * 5e-5
        line1 = {"x": loss_epoch, "y": loss, "color": self.kwards["color"][0]}
        line2 = {"x": lr_epoch, "y": lr, "color": self.kwards["color"][1]}
        # 初始化图布及子图
        fig, ax = plt.subplots(1, 2, figsize=(10, 4), dpi=300)
        # 绘制预测任务损失
        ax[0].plot(loss_epoch, loss, c=line1["color"], lw=1.5, marker="o", mfc="white", ms=5,
                   label="损失")
        plot_fillspace(ax[0], [line1, {"x": loss_epoch, "y": np.zeros_like(loss_epoch), "color": "black"}])
        ax[0].legend(loc='upper right', bbox_to_anchor=(0.98, 0.98), fontsize=15, labelcolor="black",
                     frameon=False)
        ax[0].set_xlabel("训练周期", color="black", size=20)
        ax[0].set_ylabel("损失", color="black", size=20)
        # ax[0].set_title("轨迹预测损失", color="black", size=20)
        # 绘制预测任务学习率
        ax[1].plot(line2["x"], line2["y"], c=line2["color"], lw=1.5, marker="o", mfc="white", ms=5,
                   label="学习率")
        plot_fillspace(ax[1], [line2, {"x": lr_epoch, "y": np.zeros_like(lr_epoch), "color": "black"}])
        ax[1].legend(loc='upper right', bbox_to_anchor=(0.98, 0.98), fontsize=15, labelcolor="black",
                     frameon=False)
        ax[1].set_xlabel("训练周期", color="black", size=20)
        ax[1].set_ylabel("学习率", color="black", size=20)
        # ax[1].set_title("轨迹预测学习率", color="black", size=20)
        # 所有子图参数设置
        for i in range(ax.shape[0]):
            ax[i].grid(ls="--", lw=0.5, color="black")
            ax[i].xaxis.set_tick_params(length=2, color="black", labelcolor="black", labelsize=15)
            ax[i].yaxis.set_tick_params(length=2, color="black", labelcolor="black", labelsize=15)
        plt.subplots_adjust(wspace=0.5, hspace=0.5)  # 增加宽度和高度间隔
        plt.tight_layout()  # 确保紧凑布局
        if self.kwards["save_dir"]:
            plt.savefig(f"{self.kwards['save_dir']}/pred_loss_lr.png", dpi=300, bbox_inches="tight", transparent=True)
        plt.show()

    def plot_pred_losses(self):
        """
            不同预训练方案的预测损失可视化
            参数
                log_pathes: 预测损失日志路径列表
        """
        # 初始化画布及子图
        fig, ax = plt.subplots(1, 1, figsize=(10, 6), dpi=300)
        lines = []
        for i, logpath in enumerate(self.log_pathes):
            log = event_accumulator.EventAccumulator(logpath)
            log.Reload()
            print(i, log.Tags()['scalars'])
            # continue
            loss_epoch, loss = get_data_from_log(log.Scalars("Val Loss"))
            line = {"x": loss_epoch, "y": loss, "color": self.kwards["color"][i]}
            lines.append(line)
            ax.plot(loss_epoch, loss, c=line["color"], lw=1.5, marker="o", mfc="white", ms=5,
                    label=self.kwards["legend"][i])
            ax.legend(loc='upper right', bbox_to_anchor=(0.98, 0.98), fontsize=15, labelcolor="black",
                      frameon=False)
        # plot_fillspace(ax, lines)
        # ax.set_title("轨迹预测学习率", color="black", size=20)
        # 所有子图参数设置
        ax.grid(ls="--", lw=0.5, color="black")
        ax.xaxis.set_tick_params(length=2, color="black", labelcolor="black", labelsize=15)
        ax.yaxis.set_tick_params(length=2, color="black", labelcolor="black", labelsize=15)
        ax.set_xlabel("训练周期", color="black", size=20)
        ax.set_ylabel("学习率", color="black", size=20)
        plt.subplots_adjust(wspace=0.5, hspace=0.5)  # 增加宽度和高度间隔
        plt.tight_layout()  # 确保紧凑布局
        if self.kwards["save_dir"]:
            plt.savefig(f"{self.kwards['save_dir']}/pred_losses.png", dpi=300, bbox_inches="tight", transparent=True)
        plt.show()


class MetricVizer:
    """
        指标可视化
        参数
            traj_pathes：轨迹数据路径列表
            gt_path：真实轨迹路径
            meta_path：轨迹元数据路径
            **kwargs：可选参数
    """

    def __init__(self, traj_pathes: List[str], gt_path: str, meta_path: str, **kwargs) -> None:
        self.config = load_config("../config.yaml")
        self.highd_dir = f"{self.config['common']['data_dir']}/data"
        self.highd_args = main.create_args()
        self.traj_pathes = traj_pathes
        self.gt = joblib.load(gt_path)
        self.meta = joblib.load(meta_path)
        self.kwargs = kwargs
        self.scenario_num = len(self.meta) * self.config['common']['B']

    def get_rebuild_rmse(self, traj_path: str, task: str) -> List[np.ndarray]:
        """
            获取重构轨迹的RMSE
            参数
                traj_path：重构轨迹数据路径
                task：重构任务名
            返回值
                np.array(rmse_x)：纵向RMSE数组
                np.array(rmse_y)：横向RMSE数组
                np.array(rmse_xy)：总体RMSE数组
        """
        traj = joblib.load(traj_path)
        rmse_x, rmse_y, rmse_xy = [], [], []
        metric_bar = tqdm(list(range(len(self.meta))))
        for i, (batch_gt, batch_gt_mask), batch_meta in zip(metric_bar, self.gt, self.meta):
            batch_rebuild = traj[f"{task}_Rebuild"][i].reshape(-1, 2)
            if isinstance(batch_rebuild, torch.Tensor):
                batch_rebuild = batch_rebuild.cpu().detach().numpy()
            batch_targetmask = traj[f"{task}_TargetMask"][i]
            (batch_his,) = CoordinateInverse([batch_gt[:, :, :self.config['common']['Th'], :2]], batch_meta,
                                             np.arange(len(batch_meta))).batch_inverse()
            batch_target = batch_his.cpu().detach().numpy()[~batch_targetmask]
            rmse_x.append(np.sqrt(np.mean((batch_rebuild[..., 0] - batch_target[..., 0]) ** 2)))
            rmse_y.append(np.sqrt(np.mean((batch_rebuild[..., 1] - batch_target[..., 1]) ** 2)))
            rmse_xy.append(np.sqrt(np.mean((batch_rebuild - batch_target) ** 2)))

            metric_bar.set_description(f"Processing batch: {i}")
        metric_bar.close()

        return [np.array(rmse_x), np.array(rmse_y), np.array(rmse_xy)]

    def plot_rebuild_rmse(self):
        """
            绘制重构轨迹的RMSE
        """
        RTR_RMSE = {"X": [], "Y": [], "XY": [], "Xmean": [], "Ymean": [], "XYmean": []}
        STR_RMSE = {"X": [], "Y": [], "XY": [], "Xmean": [], "Ymean": [], "XYmean": []}
        for i, traj_path in enumerate(self.traj_pathes):
            rtr_rmse_x, rtr_rmse_y, rtr_rmse_xy = self.get_rebuild_rmse(traj_path, "RTR")
            str_rmse_x, str_rmse_y, str_rmse_xy = self.get_rebuild_rmse(traj_path, "STR")
            RTR_RMSE["X"].append(rtr_rmse_x)
            RTR_RMSE["Y"].append(rtr_rmse_y)
            RTR_RMSE["XY"].append(rtr_rmse_xy)
            RTR_RMSE["Xmean"].append(rtr_rmse_x.mean())
            RTR_RMSE["Ymean"].append(rtr_rmse_y.mean())
            RTR_RMSE["XYmean"].append(rtr_rmse_xy.mean())
            STR_RMSE["X"].append(str_rmse_x)
            STR_RMSE["Y"].append(str_rmse_y)
            STR_RMSE["XY"].append(str_rmse_xy)
            STR_RMSE["Xmean"].append(str_rmse_x.mean())
            STR_RMSE["Ymean"].append(str_rmse_y.mean())
            STR_RMSE["XYmean"].append(str_rmse_xy.mean())
        # 数据维度参数
        Nbatch = len(self.meta)
        Nlegend = len(self.kwargs["legend"])
        Nxlabel = 3
        # 构造RTR数据表格
        rtr_data = {"xlabel": np.array([]), "value": np.array([]), "legend": np.array([])}
        rtr_data["xlabel"] = ["X"] * Nbatch * Nlegend + ["Y"] * Nbatch * Nlegend + ["XY"] * Nbatch * Nlegend
        rtr_data["value"] = np.concatenate(RTR_RMSE["X"] + RTR_RMSE["Y"] + RTR_RMSE["XY"], axis=0)
        rtr_data["legend"] = []
        for lgd in self.kwargs["legend"]:
            rtr_data["legend"] += [lgd] * Nbatch
        rtr_data["legend"] *= Nxlabel
        rtr_data = pd.DataFrame(rtr_data)
        # 构造STR数据表格
        str_data = {"xlabel": np.array([]), "value": np.array([]), "legend": np.array([])}
        str_data["xlabel"] = ["X"] * Nbatch * Nlegend + ["Y"] * Nbatch * Nlegend + ["XY"] * Nbatch * Nlegend
        str_data["value"] = np.concatenate(STR_RMSE["X"] + STR_RMSE["Y"] + STR_RMSE["XY"], axis=0)
        str_data["legend"] = []
        for lgd in self.kwargs["legend"]:
            str_data["legend"] += [lgd] * Nbatch
        str_data["legend"] *= Nxlabel
        str_data = pd.DataFrame(str_data)
        # 初始化图布
        fig, ax = plt.subplots(1, 2, figsize=(12, 6), dpi=300)
        # 绘制RTR任务RMSE图
        sns.barplot(ax=ax[0], data=rtr_data, x="xlabel", y="value", hue="legend", palette=self.kwargs["color"][0])
        # ax[0].set_title("RTR任务", color="black", size=20)
        # 绘制STR任务RMSE图
        sns.barplot(ax=ax[1], data=str_data, x="xlabel", y="value", hue="legend", palette=self.kwargs["color"][1])
        # ax[1].set_title("STR任务", color="black", size=20)
        # 绘制公共部分
        for i in range(ax.shape[0]):
            if self.kwargs["plotfig"][i] == 0:
                fig.delaxes(ax[i])
                continue
            ax[i].legend(loc='upper right', fontsize=15, labelcolor="black", frameon=False)
            # ax[i].grid(ls="--", lw=0.5, color="black")
            ax[i].set_xlabel("坐标轴", color="black", size=20)
            ax[i].set_ylabel("RMSE(m)", color="black", size=20)
            ax[i].set_ylim(min(ax[0].get_ylim()[0], ax[0].get_ylim()[1]), max(ax[0].get_ylim()[1], ax[1].get_ylim()[1]))
            ax[i].yaxis.set_major_locator(MultipleLocator(0.5))
            ax[i].xaxis.set_tick_params(length=2, color="black", labelcolor="black", labelsize=15)
            ax[i].yaxis.set_tick_params(length=2, color="black", labelcolor="black", labelsize=15)

        plt.subplots_adjust(wspace=0.5, hspace=0.5)  # 增加宽度和高度间隔
        plt.tight_layout()  # 确保紧凑布局
        if self.kwargs["savedir"]:
            plt.savefig(f"{self.kwargs['savedir']}/rebuild_rmse.png", dpi=300, bbox_inches="tight", transparent=True)
        plt.show()

    def get_pred_rmse(self, traj_path: str) -> List[List[np.ndarray]]:
        """
            获取预测轨迹的RMSE
            参数
                traj_path: 预测轨迹数据的恶露静
            返回值
                rmse_x：纵向预测RMSE
                rmse_y：横向预测RMSE
                rmse_xy：总体预测RMSE
        """
        rmse_x = [[], [], [], [], []]
        rmse_y = [[], [], [], [], []]
        rmse_xy = [[], [], [], [], []]
        metric_bar = tqdm(list(range(len(self.meta))))
        traj = joblib.load(traj_path)
        for i, (batch_gt, _), batch_meta in zip(metric_bar, self.gt, self.meta):
            batch_pred = traj["Pred"][i]
            (batch_target,) = CoordinateInverse([batch_gt[:, 0, self.config['common']['Th']:, :2]], batch_meta,
                                                np.arange(len(batch_meta))).batch_inverse()
            batch_target = batch_target.cpu().detach().numpy()
            for t in range(1, 6):
                rmse_x[t - 1].append(np.sqrt(np.mean((batch_pred[:, :t * 5, 0] - batch_target[:, :t * 5, 0]) ** 2)))
                rmse_y[t - 1].append(np.sqrt(np.mean((batch_pred[:, :t * 5, 1] - batch_target[:, :t * 5, 1]) ** 2)))
                rmse_xy[t - 1].append(np.sqrt(np.mean((batch_pred[:, :t * 5, :] - batch_target[:, :t * 5, :]) ** 2)))
            metric_bar.set_description(f"Processing batch: {i}")
        metric_bar.close()
        for i in range(len(rmse_x)):
            rmse_x[i] = np.array(rmse_x[i])
            rmse_y[i] = np.array(rmse_y[i])
            rmse_xy[i] = np.array(rmse_xy[i])

        return [rmse_x, rmse_y, rmse_xy]

    def plot_pred_rmse(self):
        """
            绘制预测轨迹的RMSE
        """
        RMSE_X = {1: [], 2: [], 3: [], 4: [], 5: []}
        RMSE_Y = {1: [], 2: [], 3: [], 4: [], 5: []}
        RMSE_XY = {1: [], 2: [], 3: [], 4: [], 5: []}
        for i, traj_path in enumerate(self.traj_pathes):
            rmse_x, rmse_y, rmse_xy = self.get_pred_rmse(traj_path)
            for j in range(1, 6):
                RMSE_X[j].append(rmse_x[j - 1])
                RMSE_Y[j].append(rmse_y[j - 1])
                RMSE_XY[j].append(rmse_xy[j - 1])
        for key in RMSE_X.keys():
            RMSE_X[key] = np.concatenate(RMSE_X[key], axis=0)
            RMSE_Y[key] = np.concatenate(RMSE_Y[key], axis=0)
            RMSE_XY[key] = np.concatenate(RMSE_XY[key], axis=0)
        # 数据维度参数
        Nbatch = len(self.meta)
        Nlegend = len(self.kwargs["legend"])
        Nxlabel = 5
        # 构造RMSE_X数据表格
        data_x = {"xlabel": [], "value": [], "legend": []}
        for i in range(1, 6):
            data_x["xlabel"] += [i] * Nbatch * Nlegend
            data_x["value"].append(RMSE_X[i])
        data_x["value"] = np.concatenate(data_x["value"], axis=0)
        for lgd in self.kwargs["legend"]:
            data_x["legend"] += [lgd] * Nbatch
        data_x["legend"] *= Nxlabel
        data_x = pd.DataFrame(data_x)
        # 构造RMSE_Y数据表格
        data_y = {"xlabel": [], "value": [], "legend": []}
        for i in range(1, 6):
            data_y["xlabel"] += [i] * Nbatch * Nlegend
            data_y["value"].append(RMSE_Y[i])
        data_y["value"] = np.concatenate(data_y["value"], axis=0)
        for lgd in self.kwargs["legend"]:
            data_y["legend"] += [lgd] * Nbatch
        data_y["legend"] *= Nxlabel
        data_y = pd.DataFrame(data_y)
        # 构造RMSE_XY数据表格
        data_xy = {"xlabel": [], "value": [], "legend": []}
        for i in range(1, 6):
            data_xy["xlabel"] += [i] * Nbatch * Nlegend
            data_xy["value"].append(RMSE_XY[i])
        data_xy["value"] = np.concatenate(data_xy["value"], axis=0)
        for lgd in self.kwargs["legend"]:
            data_xy["legend"] += [lgd] * Nbatch
        data_xy["legend"] *= Nxlabel
        data_xy = pd.DataFrame(data_xy)
        # 初始化图布
        fig, ax = plt.subplots(1, 3, figsize=(15, 5), dpi=300)
        # 绘制RMSE_X图
        sns.barplot(ax=ax[0], data=data_x, x="xlabel", y="value", hue="legend", palette=self.kwargs["color"][0])
        # ax[0].set_title("X方向", color="black", size=20)
        # 绘制RMSE_Y图
        sns.barplot(ax=ax[1], data=data_y, x="xlabel", y="value", hue="legend", palette=self.kwargs["color"][1])
        # ax[1].set_title("Y方向", color="black", size=20)
        # 绘制RMSE_XY图
        sns.barplot(ax=ax[2], data=data_xy, x="xlabel", y="value", hue="legend", palette=self.kwargs["color"][2])
        # ax[2].set_title("XY方向", color="black", size=20)
        # 绘制公共部分
        ymax = max(ax[0].get_ylim()[1], ax[1].get_ylim()[1], ax[2].get_ylim()[1])  # 计算纵坐标上限
        for i in range(ax.shape[0]):
            if self.kwargs["plotfig"][i] == 0:
                fig.delaxes(ax[i])
                continue
            ax[i].legend(loc='upper right', fontsize=15, labelcolor="black", frameon=False)
            # ax[i].grid(ls="--", lw=0.25, color="black")
            ax[i].set_ylim(0, ymax + 1.5)
            ax[i].yaxis.set_major_locator(MultipleLocator(0.5))
            ax[i].xaxis.set_tick_params(length=2, color="black", labelcolor="black", labelsize=15)
            ax[i].yaxis.set_tick_params(length=2, color="black", labelcolor="black", labelsize=15)
            ax[i].set_xlabel("预测时域(s)", color="black", size=20)
            ax[i].set_ylabel("RMSE(m)", color="black", size=20)
        plt.subplots_adjust(wspace=0.5, hspace=0.5)  # 增加宽度和高度间隔
        plt.tight_layout()  # 确保紧凑布局
        if self.kwargs["savedir"]:
            plt.savefig(f"{self.kwargs['savedir']}/pred_rmse.png", dpi=300, bbox_inches="tight", transparent=True)
        plt.show()


class KinematicVizer:
    """
        运动学约束相关可视化
        参数
            prob_path: 蒙特卡洛模拟概率的数据路径
            kinematic_rmse: 运动学模型的预测RMSE
            network_rmse: 网络模型的预测RMSE
            **kwargs：可选参数
    """
    def __init__(self, prob_path: str, kinematic_rmse: np.ndarray, network_rmse: np.ndarray, **kwargs):
        self.sim_prob = joblib.load(prob_path)
        self.k_rmse = kinematic_rmse
        self.n_rmse = network_rmse
        self.kwargs = kwargs

    def plot_kinematic(self, ):
        """
           绘制运动学和网络模型的RMSE，绘制运动学约束的适用概率
        """
        # 构造运动学和数据驱动方法的RMSE表格
        rmse_data = {"xlabel": [], "value": [], "legend": []}
        for t in range(1, 6):
            rmse_data["xlabel"] += [t] * 2
            rmse_data["value"].append(self.k_rmse[t - 1])
            rmse_data["value"].append(self.n_rmse[t - 1])
            rmse_data["legend"] += self.kwargs["legend"][0]
        rmse_data = pd.DataFrame(rmse_data)
        # 构造蒙特卡洛模拟的概率数据表格
        prob_data = {"xlabel": [], "value": [], "legend": []}
        for r, p in self.sim_prob.items():
            Nvalue = len(p)
            prob_data["xlabel"] += [r] * Nvalue
            prob_data["value"].append(p)
            prob_data["legend"] += self.kwargs["legend"][1] * Nvalue
        prob_data["value"] = np.concatenate(prob_data["value"], axis=0)
        prob_data = pd.DataFrame(prob_data)
        # 初始化画布和子图
        fig, ax = plt.subplots(1, 2, figsize=(12, 5), dpi=300)
        # 绘制运动学模型和深度网络预测RMSE图像
        sns.barplot(ax=ax[0], data=rmse_data, x="xlabel", y="value", hue="legend", palette=self.kwargs["color"][0])
        for i, patch in enumerate(ax[0].patches):
            x = patch.get_x()
            y = patch.get_height()
            w = patch.get_width()
            if 0 <= i < 5:
                ax[0].annotate(f'{self.k_rmse[i]:.2f}', (x + w / 2, y), ha='center', va='bottom', fontsize=12,
                               color='black')
            elif 5 <= i < 10:
                ax[0].annotate(f'{self.n_rmse[i - 5]:.2f}', (x + w / 2, y), ha='center', va='bottom', fontsize=12,
                               color='black')
        ax[0].set_xlabel("预测时域(s)", color="k", size=20)
        ax[0].set_ylabel("RMSE(m)", color="k", size=20)
        ax[0].set_ylim(0, rmse_data["value"].max() + 5)
        # ax[0].set_title("多时域下的预测均方根误差", color="k", size=20)
        # 绘制蒙特卡洛模拟概率图像
        sns.boxplot(ax=ax[1], data=prob_data, x="xlabel", y="value", hue="legend", palette=self.kwargs["color"][1],
                    boxprops=dict(linewidth=1.5),
                    whiskerprops=dict(linewidth=1.5),
                    medianprops=dict(linewidth=1.5),
                    flierprops=dict(marker='D', markersize=5, linewidth=1.5))
        ax[1].set_xlabel(r'掩码比例 $\alpha$', color="k", size=20)
        ax[1].set_ylabel("概率", color="k", size=20)
        # ax[1].set_title("蒙特卡洛模拟的平均概率", color="k", size=20)
        # 绘制公共部分
        for i in range(ax.shape[0]):
            if self.kwargs["plotfig"][i] == 0:
                fig.delaxes(ax[i])
                continue
            ax[i].legend(loc='upper left', bbox_to_anchor=(0, 0.95), fontsize=20, labelcolor="k", frameon=False)
            ax[i].xaxis.set_tick_params(length=2, color="k", labelcolor="k", labelsize=15)
            ax[i].yaxis.set_tick_params(length=2, color="k", labelcolor="k", labelsize=15)
        plt.subplots_adjust(wspace=0.5, hspace=0.5)  # 增加宽度和高度间隔
        plt.tight_layout()  # 确保紧凑布局
        if self.kwargs["savedir"]:
            plt.savefig(f"{self.kwargs['savedir']}/kinematic.png", dpi=300, bbox_inches="tight", transparent=True)
        plt.show()


class PruningVizer:
    """
        剪枝权重可视化
        参数
            ts_para_path: 时空网络的权重参数路径
            social_para_path: 交互网络的权重参数路径
    """
    def __init__(self, ts_para_path: str, social_para_path: str):
        self.ts_para = joblib.load(ts_para_path)
        self.social_para = joblib.load(social_para_path)

    def plot_ts_weight(self, level: int = 2, **kwargs):
        """
            绘制时空网络权重参数
            参数
                level: 时空网络的层级
                **kwargs：可选参数
        """
        ts_weight = self.ts_para[level]["weight"]
        ts_mask = self.ts_para[level]["unstructured_mask"]
        fig, axes = plt.subplots(2, 3, figsize=(12, 6), dpi=300)
        for i, (weight, mask) in enumerate(zip(ts_weight, ts_mask)):
            weight = np.abs(weight)
            sns.heatmap(weight, ax=axes[0, i], annot=False, cmap='viridis', xticklabels=50, yticklabels=50)
            sns.heatmap(weight, mask=~mask, ax=axes[1, i], annot=False, cmap='viridis', xticklabels=50, yticklabels=50)
        for raw in axes:
            for ax in raw:
                # ax.set_title(f"Level {level+1}, Dimension {i+1}", color="black", size=20)
                ax.legend(loc='upper right', fontsize=15, labelcolor="black", frameon=False)
                ax.xaxis.set_tick_params(length=2, color="black", labelcolor="black", labelsize=12)
                ax.yaxis.set_tick_params(length=2, color="black", labelcolor="black", labelsize=12)
                ax.set_xlabel("编码维度", color="black", size=15)
                ax.set_ylabel("输入维度", color="black", size=15)
        plt.subplots_adjust(wspace=0.5, hspace=0.5)  # 增加宽度和高度间隔
        plt.tight_layout()  # 确保紧凑布局
        if kwargs["savedir"]:
            plt.savefig(f"{kwargs['savedir']}/ts_weight.png", dpi=300, bbox_inches="tight", transparent=True)
        plt.show()

    def plot_social_weight(self, level: int = 1, **kwargs):
        """
            绘制交互网络权重参数
            参数
                level: 交互网络的层级
                **kwargs：可选参数
        """
        social_weight = self.social_para[level]["weight"]
        social_mask = self.social_para[level]["structured_mask"]
        fig, axes = plt.subplots(2, 3, figsize=(12, 6), dpi=300)
        for i, (weight, mask) in enumerate(zip(social_weight, social_mask)):
            weight = np.abs(weight)
            sns.heatmap(weight, ax=axes[0, i], annot=False, cmap='viridis', xticklabels=50, yticklabels=50)
            sns.heatmap(weight, mask=~mask, ax=axes[1, i], annot=False, cmap='viridis', xticklabels=50, yticklabels=50)
        for raw in axes:
            for ax in raw:
                ax.legend(loc='upper right', fontsize=15, labelcolor="black", frameon=False)
                colorbar = ax.collections[0].colorbar
                colorbar.ax.tick_params(labelsize=12)
                ax.xaxis.set_tick_params(length=2, color="black", labelcolor="black", labelsize=12)
                ax.yaxis.set_tick_params(length=2, color="black", labelcolor="black", labelsize=12)
                ax.set_xlabel("编码维度", color="black", size=15)
                ax.set_ylabel("输入维度", color="black", size=15)
        plt.subplots_adjust(wspace=0.5, hspace=0.5)  # 增加宽度和高度间隔
        plt.tight_layout()  # 确保紧凑布局
        if kwargs["savedir"]:
            plt.savefig(f"{kwargs['savedir']}/social_weight.png", dpi=300, bbox_inches="tight", transparent=True)
        plt.show()


if __name__ == "__main__":
    # 预测轨迹可视化
    # 左向直行:8405,4611,10627,7045,10717,11766,2186,1593左向换道:7735,2158,848,6966,7940
    # 右向直行:6317,4047,5509,2859,9667,5770,2306,288,11287右向换道:5318,2902,11200,8778,12586,4405
    # pred_vizer = PredTrajVizer("../log/pred/11/2024-12-22-15_39_37_k_dwa/test_traj.pkl",
    #                            "../datasets/highd/pred/test_meta.pkl",
    #                            None)
    # pred_vizer.plot_scenarios(scenario_ids=np.array([8405, 288, 7940, 8778]))
    # 重构轨迹可视化
    # rebuild_vizer = RebuildTrajVizer("../log/rebuild/11/2024-12-16-17_08_46_k_dwa/test_traj.pkl",
    #                                  "../datasets/highd/rebuild/val_meta.pkl",
    #                                  None)
    # # [6851,5745,11058,13584,9127,3203,2168,11887]
    # rebuild_vizer.plot_scenarios(scenario_ids=np.array([6851, 5745, 11058, 13584, 9127, 3203, 2168, 11887]), task="RTR")
    # rebuild_vizer.plot_scenarios(scenario_ids=np.array([6851, 5745, 11058, 13584, 9127, 3203, 2168, 11887]), task="STR")
    # 重构损失可视化
    rebuild_log_vizer = RebuildLogVizer("C:\\Users\\Lenovo\\Desktop")
    rebuild_log_vizer.plot_rebuild_ax1_two(
        ["E:\\download\\韩帅_硕士期间资料备份\\Project\\TrajPred\\MGS-Traj+\\log\\rebuild\\11\\2024-12-16-17_08_46_k_dwa\\events.out.tfevents.1734340188.DESKTOP-6O3C2CV.20104",
         "E:\\download\\韩帅_硕士期间资料备份\\Project\\TrajPred\\MGS-Traj+\\log\\rebuild\\11\\2024-12-24-16_52_19_nok\\events.out.tfevents.1735030410.DESKTOP-6O3C2CV.16068"]
    )
    # 重构损失权重及学习率可视化
    # rebuild_log_vizer.plot_rebuild_weight_lr(
    #     ["../log/rebuild/11/2025-01-04-13_24_59_nodwa/events.out.tfevents.1735968362.DESKTOP-6O3C2CV.13832.0",
    #      "../log/rebuild/11/2025-01-04-13_24_59_nodwa/events.out.tfevents.1735968362.DESKTOP-6O3C2CV.13832.1"])
    # 预测损失和学习率可视化
    # pred_log_vizer1 = PredLogVizer(
    #     ["../log/pred/11/2024-12-22-15_39_37_k_dwa/events.out.tfevents.1734853268.DESKTOP-6O3C2CV.20560.1",
    #      "../log/pred/11/2024-12-22-15_39_37_k_dwa/events.out.tfevents.1734853268.DESKTOP-6O3C2CV.20560.0"],
    #     color=["darkorchid", "darkorchid"], save_dir="E:/OneDrive/最近/毕业论文/毕业答辩/插图")
    # pred_log_vizer1.plot_pred_loss_lr()
    # 多种预测损失可视化
    # pred_log_vizer2 = PredLogVizer(
    #     ["../log/pred/00/2024-12-16-17_09_07/events.out.tfevents.1734340228.DESKTOP-6O3C2CV.6420.1",
    #      "../log/pred/10/2024-12-24-15_58_07_k_dwa/events.out.tfevents.1735027161.DESKTOP-6O3C2CV.8040.1",
    #      "../log/pred/01/2024-12-24-15_58_49_k_dwa/events.out.tfevents.1735027204.DESKTOP-6O3C2CV.2344.1",
    #      "../log/pred/11/2024-12-22-15_39_37_k_dwa/events.out.tfevents.1734853268.DESKTOP-6O3C2CV.20560.1"],
    #     legend=["TPNet(从头训练)", "TPNet(RTR)", "TPNet(STR)", "TPNet(RTR+STR)"],
    #     color=["skyblue", "royalblue", "plum", "darkorchid"],
    #     save_dir="E:/OneDrive/最近/毕业论文/毕业答辩/插图")
    # pred_log_vizer2.plot_pred_losses()
    # 重构RMSE可视化
    # rebuild_metric_vizer = MetricVizer(["../log/rebuild/11/2024-12-16-17_08_46_k_dwa/test_traj.pkl"],
    #                                    "../datasets/highd/rebuild/val_loader.pkl.",
    #                                    "../datasets/highd/rebuild/val_meta.pkl",
    #                                    legend=["TRNet"],
    #                                    color=[["cornflowerblue"], ["cornflowerblue"]],
    #                                    plotfig=[1, 1],
    #                                    savedir="E:/OneDrive/最近/毕业论文/毕业答辩/插图")
    # rebuild_metric_vizer.plot_rebuild_rmse()
    # 预测RMSE可视化
    # pred_rmse_vizer = MetricVizer(["../log/pred/00/2024-12-16-17_09_07/test_traj.pkl",
    #                                "../log/pred/10/2024-12-24-15_58_07_k_dwa/test_traj.pkl",
    #                                "../log/pred/01/2024-12-24-15_58_49_k_dwa/test_traj.pkl",
    #                                "../log/pred/11/2024-12-22-15_39_37_k_dwa/test_traj.pkl"],
    #                               "../datasets/highd/pred/test_loader.pkl.",
    #                               "../datasets/highd/pred/test_meta.pkl",
    #                               legend=["TPNet(从头训练)", "TPNet(RTR)", "TPNet(STR)", "TPNet(RTR+STR)"],
    #                               color=[["skyblue", "royalblue", "plum", "darkorchid"],
    #                                      ["skyblue", "royalblue", "plum", "darkorchid"],
    #                                      ["skyblue", "royalblue", "plum", "darkorchid"]],
    #                               plotfig=[1, 1, 1],
    #                               savedir="E:/OneDrive/最近/毕业论文/毕业答辩/插图")
    # pred_rmse_vizer.plot_pred_rmse()

    # 运动学可行性验证可视化
    # kinematic_vizer = KinematicVizer("../log/kinematic/2025-01-11-15_45_15/sim_prob.pkl",
    #                                  np.array([0.027253, 14.242, 16.4367, 33.3652, 40.2207]),
    #                                  np.array([0.519839, 0.544603, 0.593737, 0.677918, 0.847814]),
    #                                  color=[["cornflowerblue", "indianred"], ["cornflowerblue"]],
    #                                  legend=[["运动学模型", "网络模型"], ["概率"]],
    #                                  plotfig=[1, 1],
    #                                  savedir="E:/OneDrive/最近/毕业论文/毕业答辩/插图")
    # kinematic_vizer.plot_kinematic()

    # 剪枝权重参数可视化
    # prune_vizer = PruningVizer("../log/pruning/2025-02-21-15_36_20_0.1_0.1/ts_para.pkl",
    #                            "../log/pruning/2025-02-21-15_36_20_0.1_0.1/social_para.pkl")
    # prune_vizer.plot_ts_weight(savedir="E:/OneDrive/最近/毕业论文/毕业答辩/插图")
    # prune_vizer.plot_social_weight(savedir="E:/OneDrive/最近/毕业论文/毕业答辩/插图")
