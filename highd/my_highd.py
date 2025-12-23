import os

import joblib
import numpy as np
import torch
import torch.nn.utils.rnn as rnn_utils
import torch.utils.data as data
from einops import rearrange
from scipy.interpolate import interp1d
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

from config import *
from highd.api import main, read_csv


class Scenario:
    """
        交互式场景建模，对原始highd数据在车辆、轨迹、特征级别依次进行处理，重新构建自动驾驶场景
        参数
            record_id：60条记录的id，1~60
            config：参数设置
    """

    def __init__(self, record_id, config):
        self.record_id = record_id
        self.config = config
        self.record_dir = f"{self.config['common']['data_dir']}/data"
        self.point_num = (self.config['common']['his_horizon'] + self.config['common']['fut_horizon']) * \
                         self.config['common']['sampling_frequency']

    def read_records(self):
        """
            读取60个record，并清洗掉无效信息
            返回值
                records：包含60个record的字典，key为record id，value为record数据
        """
        highd_args = main.create_args()
        records = {rid: {} for rid in self.record_id}
        read_bar = tqdm(list(self.record_id - 1))
        for _, rid in zip(read_bar, self.record_id):
            rid_str = str(rid).zfill(2)
            highd_args["input_path"] = f"{self.record_dir}/{rid_str}_tracks.csv"
            highd_args["input_static_path"] = f"{self.record_dir}/{rid_str}_tracksMeta.csv"
            highd_args["input_meta_path"] = f"{self.record_dir}/{rid_str}_recordingMeta.csv"
            tracks = read_csv.read_track_csv(highd_args)
            track_metas = read_csv.read_static_info(highd_args)
            record_meta = read_csv.read_meta_info(highd_args)
            record = {tid: {} for tid in range(1, len(tracks) + 1)}
            for track, (track_id, track_meta) in zip(tracks, track_metas.items()):
                valid_track_data = {"trackID": track_id}
                valid_track_data.update(
                    {
                        key: value
                        for key, value in track.items()
                        if key
                           # 这些是轨迹数据中需要过滤掉的部分
                           not in [
                               "frontSightDistance",
                               "backSightDistance",
                               "dhw",
                               "thw",
                               "ttc",
                               "precedingXVelocity",
                               "id",
                           ]
                    }
                )
                valid_track_data.update(
                    {
                        key: value
                        for key, value in track_meta.items()
                        if key
                           # 这些是轨迹元数据中的有效部分
                           in [
                               "class",
                               "drivingDirection",
                               "numLaneChanges",
                           ]
                    }
                )
                valid_track_data.update(
                    {
                        key: value
                        for key, value in record_meta.items()
                        if key
                           # 这些是record元数据中的有效部分
                           in ["locationId", "upperLaneMarkings", "lowerLaneMarkings"]
                    }
                )
                valid_track_data.update({"recordID": record_meta["id"]})
                record[track_id] = valid_track_data
            records[rid] = record

            read_bar.set_description(f"reading records: {rid}/{len(read_bar)}")

        return records

    def build_scenarios(self):
        """
            构建多个交互式场景
            返回值
                scenarios：场景数据，是一个包含多个字典的列表，每个字典都是一个独立的场景，包括场景中的轨迹和元数据
        """
        records = self.read_records()
        scenario_bar = tqdm(list(range(len(records))))
        scenarios = []
        for i, (record_id, record) in zip(scenario_bar, records.items()):
            for track_id, track in record.items():
                Ltv = len(track["frame"])
                if Ltv < self.point_num:
                    # 确保目标车辆轨迹长度符合要求
                    continue
                else:
                    # 获取目标车辆的周围车辆id
                    sv_ids = np.concatenate(
                        (
                            track["leftPrecedingId"].reshape(-1, 1),
                            track["leftAlongsideId"].reshape(-1, 1),
                            track["leftFollowingId"].reshape(-1, 1),
                            track["precedingId"].reshape(-1, 1),
                            track["followingId"].reshape(-1, 1),
                            track["rightPrecedingId"].reshape(-1, 1),
                            track["rightAlongsideId"].reshape(-1, 1),
                            track["rightFollowingId"].reshape(-1, 1),
                        ),
                        axis=-1,
                    )
                    # 构建目标车辆轨迹的索引字典，包括目标车辆id，轨迹序列索引，所在车道id，换道次数
                    tv_indices = dict(id=track_id, index=np.arange(Ltv), laneIDs=track["laneId"],
                                      laneChangeNum=track["numLaneChanges"])
                    # 裁剪目标车辆轨迹序列的索引
                    tv_indices = self.cut_tv_indices(
                        tv_indices,
                        (self.config['common']['his_horizon'] + self.config['common']['fut_horizon']) *
                        self.config['common']['sampling_frequency'],
                        (self.config['common']['his_horizon'] + self.config['common']['fut_horizon']) *
                        self.config['common']['sampling_frequency']
                    )
                    # 按照目标车辆轨迹裁剪周围车辆轨迹，选取特征并构建场景
                    for tv_id, tv_idx in tv_indices.items():
                        idx = tv_idx["index"]
                        sv_id = sv_ids[idx[: self.config['common']['his_horizon'] * self.config['common'][
                            'sampling_frequency']], :]
                        sv_indices = {}
                        for k in range(sv_id.shape[1]):
                            unique_id = np.unique(sv_id[:, k])
                            unique_id = unique_id[unique_id != 0]
                            if np.size(unique_id) > 0:
                                for uid in unique_id:
                                    uid_idx = np.where(sv_id[:, k] == uid)[0]
                                    if np.size(uid_idx) >= self.config['common']['sampling_frequency'] * 1:
                                        Tsv = len(record[uid]["frame"])
                                        sv_indices.update(
                                            {
                                                uid: {
                                                    "keyIndex": uid_idx,
                                                    "wholeIndex": np.arange(Tsv),
                                                    "laneIDs": record[uid]["laneId"],
                                                    "laneChangeNum": record[uid]["numLaneChanges"]
                                                }
                                            }
                                        )
                        scenario_indices = {tv_id: tv_idx}
                        if len(sv_indices) > 0:
                            sv_indices = self.cut_sv_indices(sv_indices)
                            scenario_indices.update(sv_indices)
                        scenario_trajs, scenario_meta = self.build_one_scenario(record, scenario_indices)
                        scenarios.append({"Traj": scenario_trajs, "Meta": scenario_meta})
            scenario_bar.set_description(
                f"building scenes from records: {i + 1}/{len(records)}"
            )
        print(f"totally built {len(scenarios)} scenarios!")
        return scenarios

    def cut_tv_indices(self, tv_indices, window, step):
        """
            裁剪目标车辆轨迹索引
            参数
                tv_indices：目标车辆轨迹索引字典
                window：滑动窗口宽度
                step：滑动步长
            返回值
                cut_indices：裁剪后的目标车辆轨迹索引字典
        """
        cut_indices = dict()
        id_counter = 0.01
        if tv_indices["laneChangeNum"] > 0:
            delta_lane_ids = np.diff(tv_indices["laneIDs"])
            lc_index = np.nonzero(delta_lane_ids)[0]
            for idx in lc_index:
                cut_index = dict(index=np.array([]), lcType=np.array([]))
                tv_start, tv_end = tv_indices["index"][0], tv_indices["index"][-1]
                lc_start, lc_end = idx - self.config['common']['his_horizon'] * self.config['common'][
                    'sampling_frequency'], idx + self.config['common']['fut_horizon'] * self.config['common'][
                                       'sampling_frequency']
                delta_start, delta_end = lc_start - tv_start, tv_end - lc_end
                if delta_start >= 0 and delta_end >= 0:
                    cut_index["index"] = tv_indices["index"][lc_start:lc_end]
                elif delta_start < 0 and delta_end >= 0:
                    cut_index["index"] = tv_indices["index"][lc_start + abs(delta_start): lc_end + abs(delta_start)]
                elif delta_start >= 0 and delta_end < 0:
                    cut_index["index"] = tv_indices["index"][lc_start - abs(delta_end) + 1:lc_end - abs(delta_end) + 1]
                if delta_lane_ids[idx] < 0:
                    cut_index["lcType"] = np.array(-1)
                elif delta_lane_ids[idx] > 0:
                    cut_index["lcType"] = np.array(1)
                cut_indices.update({tv_indices["id"] + id_counter: cut_index})
                id_counter += 0.01
        else:
            for i in range(0, len(tv_indices["index"]) - window + 1, step):
                cut_index = dict(index=np.array([]), lcType=np.array([]))
                cut_index["index"] = tv_indices["index"][i: i + window]
                cut_index["lcType"] = np.array(0)
                cut_indices.update({tv_indices["id"] + id_counter: cut_index})
                id_counter += 0.01

        return cut_indices

    def cut_sv_indices(self, sv_indices):
        """
            裁剪周围车辆轨迹索引
            参数
                sv_indices：周围车辆轨迹索引字典
            返回值
                cut_indices：裁剪后的目标车辆轨迹索引字典
        """
        cut_indices = dict()
        for sv_id, sv_index in sv_indices.items():
            cut_index = dict(index=np.array([]), lcType=np.array([]))
            key_start, key_end = sv_index["keyIndex"][0], sv_index["keyIndex"][-1]
            whole_start, whole_end = sv_index["wholeIndex"][0], sv_index["wholeIndex"][-1]
            delta_lane_ids = np.diff(sv_index["laneIDs"])
            lc_index = np.nonzero(delta_lane_ids)[0]
            if key_start + self.point_num <= whole_end:
                cut_index["index"] = sv_index["wholeIndex"][key_start:key_start + self.point_num]
            else:
                delta_index = self.point_num - (whole_end - key_start + 1)
                if key_start - delta_index >= whole_start:
                    cut_index["index"] = sv_index["wholeIndex"][key_start - delta_index:whole_end + 1]
                else:
                    if len(sv_index["wholeIndex"]) >= self.config['common']['his_horizon'] * \
                            self.config['common']['sampling_frequency']:
                        cut_index["index"] = sv_index["wholeIndex"][whole_start:whole_end + 1]
            if len(cut_index["index"]) > 0:
                if sv_index["laneChangeNum"] == 0:
                    cut_index["lcType"] = np.array(0)
                elif sv_index["laneChangeNum"] > 0:
                    for idx in lc_index:
                        if delta_lane_ids[idx] < 0:
                            cut_index["lcType"] = np.array(-1)
                        elif delta_lane_ids[idx] > 0:
                            cut_index["lcType"] = np.array(1)
                cut_index["lcType"] = np.array(cut_index["lcType"])
                cut_indices.update({sv_id: cut_index})
        return cut_indices

    def build_one_scenario(self, record, indices):
        """
            构建一个交互式场景
            参数
                record：当前目标车辆及相关的周围车辆所在的record
                indices：目标车辆和周围车辆的轨迹索引字典
            返回值
                scenario_trajs：当前场景的轨迹数据
                scenario_meta：当前场景的元数据
        """
        scenario_trajs = []
        scenario_lanes = []
        scenario_meta = {
            "locationID": 0,
            "recordID": 0,
            "trackID": [],
            "laneID": [],
            "axisRange": [],
            "centerPoint": [],
            "trackFrame": [],
            "lane": [],
            "centerLine": [],
        }
        X = []
        Y = []
        for i, (traj_id, traj_idx) in enumerate(indices.items()):
            traj = record[np.floor(traj_id)]
            lc_type = traj_idx["lcType"]
            if i == 0:
                if traj["drivingDirection"] == 1:
                    Y.append(traj["upperLaneMarkings"])
                    scenario_meta["laneID"] = np.arange(1, len(traj["upperLaneMarkings"]))
                    lc_type = lc_type * (-1)
                elif traj["drivingDirection"] == 2:
                    Y.append(traj["lowerLaneMarkings"])
                    scenario_meta["laneID"] = np.arange(1, len(traj["lowerLaneMarkings"]))
                scenario_meta["recordID"] = traj["recordID"]
                scenario_meta["locationID"] = traj["locationId"]
                scenario_meta["centerPoint"] = np.array(
                    [traj["bbox"][self.config['common']['his_horizon'] * self.config['common'][
                        'sampling_frequency'] - 1, 0],
                     traj["bbox"][self.config['common']['his_horizon'] * self.config['common'][
                         'sampling_frequency'] - 1, 1]])
            scenario_meta["trackID"].append(traj_id)

            x = traj["bbox"][:, 0] + 0.5 * traj["bbox"][:, 2]
            y = traj["bbox"][:, 1] + 0.5 * traj["bbox"][:, 3]
            vx = traj["xVelocity"]
            vy = traj["yVelocity"]
            ax = traj["xAcceleration"]
            ay = traj["yAcceleration"]
            track_class = traj["class"]
            if track_class == "Car":
                track_class = np.full(len(x), fill_value=1)
            elif track_class == "Truck":
                track_class = np.full(len(x), fill_value=2)
            direction = np.full(len(x), fill_value=traj["drivingDirection"])
            lc_type = np.full(len(x), fill_value=lc_type)
            frame = traj["frame"]

            traj = np.concatenate(
                (
                    x.reshape(-1, 1),
                    y.reshape(-1, 1),
                    vx.reshape(-1, 1),
                    vy.reshape(-1, 1),
                    ax.reshape(-1, 1),
                    ay.reshape(-1, 1),
                    track_class.reshape(-1, 1),
                    direction.reshape(-1, 1),
                    lc_type.reshape(-1, 1),
                    frame.reshape(-1, 1)
                ),
                axis=-1,
            )[traj_idx["index"], :]
            scenario_meta["trackFrame"].append(frame[traj_idx["index"]])
            X.append(traj[:, 0])
            scenario_trajs.append(traj)
        X = np.concatenate(X, axis=0).reshape(-1)
        Y = np.array(Y).reshape(-1)

        scenario_meta["axisRange"] = np.array(
            [
                np.min(X),
                np.max(X),
                np.min(Y),
                np.max(Y),
            ]
        )
        scenario_meta["trackID"] = np.array(scenario_meta["trackID"])
        for y in Y:
            lane = np.array([[scenario_meta["axisRange"][0], y], [scenario_meta["axisRange"][1], y]])
            scenario_lanes.append(lane)
        scenario_trajs = self.process_trajs(scenario_trajs, scenario_meta)
        scenario_lanes, scenario_centerlines = self.process_lanes(scenario_lanes, scenario_meta)
        scenario_meta["lane"] = scenario_lanes
        scenario_meta["centerLine"] = scenario_centerlines

        return scenario_trajs, scenario_meta

    def process_trajs(self, trajs, meta):
        """
            选取特征并进行归一化、中心化和下采样处理
            参数
                trajs：当前场景的原始轨迹数据
                meta：当前场景的元数据
            返回值
                processed_trajs：经过特征处理后的轨迹数据
        """
        processed_trajs = []
        frames = np.concatenate(meta["trackFrame"], axis=0)
        frame_min, frame_max = frames.min(), frames.max()
        xmin, xmax, ymin, ymax = meta["axisRange"]
        xc, yc = meta["centerPoint"]

        for traj in trajs:
            # 中心化和归一化
            traj[:, 0] = (traj[:, 0] - xc) / (xmax - xmin)
            traj[:, 1] = (traj[:, 1] - yc) / (ymax - ymin)
            traj[:, 2] = traj[:, 2] / (xmax - xmin)
            traj[:, 3] = traj[:, 3] / (ymax - ymin)
            traj[:, 4] = traj[:, 4] / (xmax - xmin)
            traj[:, 5] = traj[:, 5] / (ymax - ymin)
            traj[:, -1] = (traj[:, -1] - frame_min) / (frame_max - frame_min)
            # 对不足标准长度的轨迹进行0填充和掩码标记
            L, D = traj.shape
            deltaL = (self.config['common']['his_horizon'] + self.config['common']['fut_horizon']) * \
                     self.config['common']['sampling_frequency'] - L
            mask = np.zeros((self.config['common']['his_horizon'] + self.config['common']['fut_horizon']) *
                            self.config['common']['sampling_frequency'])
            if deltaL > 0:
                pad = np.zeros((deltaL, D))
                traj = np.concatenate((traj, pad), axis=0)
                mask[-deltaL:] = 1
            traj = np.concatenate((traj, mask.reshape(-1, 1)), axis=-1)
            ds_traj = []
            # 5倍下采样，采样后相邻轨迹点间隔0.2s
            for i in range(L):
                if i % (self.config['common']['sampling_frequency'] // self.config['common'][
                    'downsampling_rate']) == 0:
                    ds_traj.append(traj[i, :].reshape(1, -1))
            ds_traj = np.concatenate(ds_traj, axis=0)
            processed_trajs.append(ds_traj)
        processed_trajs = np.concatenate(processed_trajs, axis=0)

        return processed_trajs

    def process_lanes(self, lanes, meta):
        """
            处理车道线，对车道线进行插值，并推算车道中心线
            参数
                lanes：车道线横坐标
                meta：当前场景的元数据
            返回值
                Lanes：处理后的车道线
                CenterLines：车道中心线
        """
        Lanes = []
        # CenterLineVecs = []
        CenterLines = []
        xmin, xmax, ymin, ymax = meta["axisRange"]
        xc, yc = meta["centerPoint"]
        lane_ids = meta["laneID"]
        for lane in lanes:
            x, y = lane[:, 0], lane[:, 1]
            dists = np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2)
            cumulative_dists = np.concatenate((np.array([0]), np.cumsum(dists)))
            total_dist = cumulative_dists[-1]
            new_dists = np.arange(0, total_dist, self.config['common']['lanepoint_interval'])
            new_dists = np.append(new_dists, total_dist)
            interp_func_x = interp1d(cumulative_dists, x, kind="linear")
            interp_func_y = interp1d(cumulative_dists, y, kind="linear")
            # new_x = (interp_func_x(new_dists) - xc) / (xmax - xmin)
            # new_y = (interp_func_y(new_dists) - yc) / (ymax - ymin)
            new_x = interp_func_x(new_dists)
            new_y = interp_func_y(new_dists)
            lane = np.concatenate((new_x.reshape(-1, 1), new_y.reshape(-1, 1)), axis=-1)
            Lanes.append(lane)
        for i in range(len(Lanes) - 1):
            center_line = (Lanes[i] + Lanes[i + 1]) / 2
            lane_id = np.full([len(center_line), 1], fill_value=lane_ids[i])
            center_line = np.concatenate((center_line, lane_id), axis=-1)
            CenterLines.append(self.pad_center_line(center_line))
            L, D = center_line.shape
            window, step = self.config['common']['lanepoint_num'], self.config['common']['lanepoint_num'] - 1
            n = (L - 1) // step
            deltaL = (n + 1) * step + 1 - L
            mask = np.array([])
            if 0 < deltaL < step:
                mask = np.zeros(L + deltaL)
                mask[-deltaL:] = 1
                pad = np.zeros((deltaL, D))
                center_line = np.concatenate((center_line, pad), axis=0)
            elif deltaL == step:
                mask = np.zeros(L)
            center_line = np.concatenate((center_line, mask.reshape(-1, 1)), axis=-1)
            L1, D1 = center_line.shape
            center_polylines = []
            for j in range(0, L1 - window + 1, step):
                center_polylines.append(center_line[j: j + window, :])
            # for line in center_polylines:
            #     for k in range(line.shape[0] - 1):
            #         x1 = line[k, 0]
            #         y1 = line[k, 1]
            #         x2 = line[k + 1, 0]
            #         y2 = line[k + 1, 1]
            #         _id_ = line[k + 1, 2]
            #         m = line[k + 1, 3]
            #         if m == 0:
            #             CenterLineVecs.append([x1, y1, x2, y2, x2 - x1, y2 - y1, _id_, m])
            #         elif m == 1:
            #             CenterLineVecs.append([0, 0, 0, 0, 0, 0, 0, m])
        # CenterLineVecs = np.array(CenterLineVecs)
        CenterLines = np.concatenate(CenterLines, axis=0)
        return Lanes, CenterLines

    def pad_center_line(self, line):
        """
            对车道中心线进行0填充，使得每段中心线序列等长
            参数
                line：车道中心线序列
            返回值
                line：填充后的车道中心线
        """
        L, D = line.shape
        mask = np.zeros(L)
        resL = L % self.config['common']['lanepoint_num']
        if resL > 0:
            pad = np.zeros((self.config['common']['lanepoint_num'] - resL, D))
            line = np.concatenate([line, pad], axis=0)
            mask = np.concatenate((mask, np.ones(self.config['common']['lanepoint_num'] - resL)), axis=0)
        line = np.concatenate([line, mask.reshape(-1, 1)], axis=-1)
        return line


class DataSet:
    """
        数据集划分类，每个场景作为一个样本，划分训练集、验证集和测试集
        参数
            scenarios：场景数据列表
            config：配置参数
    """

    def __init__(self, scenarios, config):
        self.config = config
        self.Trajs, self.TrajMasks, self.Metas = [], [], []
        for scenario in scenarios:
            self.Trajs.append(torch.as_tensor(scenario["Traj"][:, :-1], dtype=torch.float32))
            self.TrajMasks.append(torch.as_tensor(scenario["Traj"][:, -1], dtype=torch.bool))
            self.Metas.append(scenario["Meta"])
        self.batch_size = self.config['common']['B']
        self.This = self.config['common']['Th']
        self.Tfut = self.config['common']['Tf']
        self.dataloader_dir = self.config['common']['data_dir']
        if not os.path.exists(self.dataloader_dir):
            os.makedirs(self.dataloader_dir)

    def build_dataset(self):
        """
            构建通用数据集，训练：验证：测试=8：1：1
            返回值
                train_set：训练集TensorDataset对象
                train_metas：训练集元数据列表
                val_set：验证集TensorDataset对象
                val_metas：验证集元数据列表
                test_set：测试集TensorDataset对象
                test_metas：测试集元数据列表
        """
        print("Building dataset...")
        self.Trajs = rearrange(
            rnn_utils.pad_sequence(self.Trajs, batch_first=True, padding_value=0),
            "B (N T) Da -> B N T Da",
            T=self.This + self.Tfut,
        )
        self.TrajMasks = rearrange(
            rnn_utils.pad_sequence(self.TrajMasks, batch_first=True, padding_value=True),
            "B (N T) -> B N T",
            T=self.This + self.Tfut,
        )

        datazip = list(zip(self.Trajs.unsqueeze(1), self.TrajMasks.unsqueeze(1), self.Metas))
        train_data, val_data = train_test_split(datazip, test_size=0.2, random_state=42)
        val_data, test_data = train_test_split(val_data, test_size=0.5, random_state=42)
        (train_trajs, train_traj_masks, train_metas) = zip(*train_data)
        (val_trajs, val_traj_masks, val_metas) = zip(*val_data)
        (test_trajs, test_traj_masks, test_metas) = zip(*test_data)
        train_set = data.TensorDataset(
            torch.cat(train_trajs, dim=0),
            torch.cat(train_traj_masks, dim=0),
        )
        val_set = data.TensorDataset(
            torch.cat(val_trajs, dim=0),
            torch.cat(val_traj_masks, dim=0),
        )
        test_set = data.TensorDataset(
            torch.cat(test_trajs, dim=0),
            torch.cat(test_traj_masks, dim=0),
        )
        return train_set, train_metas, val_set, val_metas, test_set, test_metas

    def build_pred_dataloader(self):
        """
            构建轨迹预测数据集，训练：验证：测试=8：1：1
        """
        train_set, train_metas, val_set, val_metas, test_set, test_metas = self.build_dataset()
        train_loader = data.DataLoader(
            train_set, batch_size=self.batch_size, shuffle=False, drop_last=True
        )
        val_loader = data.DataLoader(
            val_set, batch_size=self.batch_size, shuffle=False, drop_last=True
        )
        test_loader = data.DataLoader(
            test_set, batch_size=self.batch_size, shuffle=False, drop_last=True
        )

        train_metas = self.divide_meta_bacthes(train_metas, drop_last=True)
        val_metas = self.divide_meta_bacthes(val_metas, drop_last=True)
        test_metas = self.divide_meta_bacthes(test_metas, drop_last=True)

        joblib.dump(
            train_loader, f"{self.dataloader_dir}/pred/train_loader.pkl"
        )
        joblib.dump(train_metas, f"{self.dataloader_dir}/pred/train_meta.pkl")
        joblib.dump(val_loader, f"{self.dataloader_dir}/pred/val_loader.pkl")
        joblib.dump(val_metas, f"{self.dataloader_dir}/pred/val_meta.pkl")
        joblib.dump(test_loader, f"{self.dataloader_dir}/pred/test_loader.pkl")
        joblib.dump(test_metas, f"{self.dataloader_dir}/pred/test_meta.pkl")

        print("Done!")

    def build_rebuild_dataloader(self):
        """
            构建轨迹重构数据集，训练：验证=9：1
        """
        print("Building traj rebuild dataloader...")
        train_set, train_metas, val_set, val_metas, test_set, test_metas = self.build_dataset()
        train_set = data.ConcatDataset([train_set, val_set])
        train_metas += val_metas
        val_set = test_set
        val_metas = test_metas
        train_loader = data.DataLoader(
            train_set, batch_size=self.batch_size, shuffle=False, drop_last=True
        )
        val_loader = data.DataLoader(
            val_set, batch_size=self.batch_size, shuffle=False, drop_last=True
        )
        train_metas = self.divide_meta_bacthes(train_metas, drop_last=True)
        val_metas = self.divide_meta_bacthes(val_metas, drop_last=True)
        joblib.dump(train_loader, f"{self.dataloader_dir}/rebuild/train_loader.pkl")
        joblib.dump(train_metas, f"{self.dataloader_dir}/rebuild/train_meta.pkl")
        joblib.dump(val_loader, f"{self.dataloader_dir}/rebuild/val_loader.pkl")
        joblib.dump(val_metas, f"{self.dataloader_dir}/rebuild/val_meta.pkl")

        print("Done!")

    def divide_meta_bacthes(self, meta, drop_last):
        """
            按批量划分元数据
            参数
                meta：元数据列表
                drop_last：标志位，如果为1且最后一个批量的大小不足128,则丢该批量的元数据
            返回值
                metaBatches：按批量重新划分的元数据多级列表，第一个维度为批量数，第二个维度为单个批量的大小，即128
        """
        metaBatches = []
        for i in range(0, len(meta), self.batch_size):
            batch = meta[i: i + self.batch_size]
            metaBatches.append(batch)
        if drop_last == True and len(metaBatches[-1]) < self.batch_size:
            metaBatches.pop()
        return metaBatches


if __name__ == "__main__":
    config = load_config("../config.yaml")
    scenarios = Scenario(np.arange(1, 61), config).build_scenarios()
    DataSet(scenarios, config).build_pred_dataloader()
    DataSet(scenarios, config).build_rebuild_dataloader()
