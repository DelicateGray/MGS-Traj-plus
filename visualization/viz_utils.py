from typing import List, Dict, Any

import imageio.v2 as imageio
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes


def get_sa_id(ta_track: dict, last_his_frame: int) -> np.ndarray:
    sa_id = np.concatenate(
        (
            ta_track["leftPrecedingId"].reshape(-1, 1),
            ta_track["leftAlongsideId"].reshape(-1, 1),
            ta_track["leftFollowingId"].reshape(-1, 1),
            ta_track["precedingId"].reshape(-1, 1),
            ta_track["followingId"].reshape(-1, 1),
            ta_track["rightPrecedingId"].reshape(-1, 1),
            ta_track["rightAlongsideId"].reshape(-1, 1),
            ta_track["rightFollowingId"].reshape(-1, 1),
        ),
        axis=-1,
    )[np.isin(ta_track["frame"], last_his_frame)]

    return sa_id[sa_id != 0]


def get_scenario_data(traj: dict, meta: [dict], scenario_id: int, task: str = "RTR", batch_size: int = 128) -> dict:
    b = (scenario_id + 1) // batch_size
    n = (scenario_id + 1) % batch_size - 1
    B_idx = traj[f"{task}_MetaIndex"][b]
    (_, N_idx) = np.unique(B_idx, return_index=True)
    N_idx = np.concatenate([N_idx, np.array([len(B_idx)])])
    n_start, n_end = N_idx[n], N_idx[n + 1]
    n_meta = traj[f"{task}_MetaIndex"][b][n_start]
    meta = meta[b][n_meta]
    scenario_data = {"Rebuild": traj[f"{task}_Rebuild"][b][n_start:n_end, ...],
                     "HisMask": traj[f"{task}_TargetMask"][b][n, ...],
                     "Meta": meta}
    if task == "RTR":
        scenario_data.update(
            {"Kinematic": traj[f"{task}_Kinematic"][b][n_start:n_end, ...]})

    return scenario_data


def divide_mask_segment(vehicle_mask: np.ndarray) -> list:
    # T = len(vehicle_mask)
    # mask_index = np.where(vehicle_mask == 0)[0]
    unmask_index = np.where(vehicle_mask == 1)[0]
    mask_segment, unmask_segment = [], []
    start = 0
    while start < len(unmask_index):
        cur_seg = [unmask_index[start]]  # ��ʼһ��
        while start + 1 < len(unmask_index) and unmask_index[start + 1] == unmask_index[start] + 1:
            cur_seg.append(unmask_index[start + 1])  # �����������
            start += 1  # �ƶ�ָ��
        unmask_segment.append(np.array(cur_seg))  # ���浱ǰ��
        start += 1  # �ƶ�����һ��Ԫ��
    return unmask_segment


def get_data_from_log(scalar: List[Dict[str, Any]]) -> np.ndarray:
    step = []
    value = []

    for point in scalar:
        step.append(point.step)
        value.append(point.value)

    return np.array(step), np.array(value)


def plot_bbox(bbox: np.ndarray[float], style: dict) -> patches.Rectangle:
    return patches.Rectangle(
        (bbox[0], bbox[1]),
        bbox[2],
        bbox[3],
        **style,
    )


def plot_triangle(triangle: np.ndarray[float], style: dict) -> patches.Polygon:
    return patches.Polygon(triangle, closed=True, **style, )


def plot_line(ax: Axes, traj: np.ndarray, style: dict) -> [plt.Line2D]:
    return ax.plot(
        traj[:, 0],
        traj[:, 1],
        **style,
    )


def plot_point(ax: Axes, traj: np.ndarray, style: dict):
    return ax.scatter(
        traj[:, 0],
        traj[:, 1],
        **style,
    )


# def plot_fillspace(ax: Axes, line1: Dict[str, Any], line2: Dict[str, Any], fill_between_points: int = 9) -> None:
#     x = line1["x"]
#     y1, color1 = line1["y"], line1["color"]
#     y2, color2 = line2["y"], line2["color"]
#     l = len(line1["x"])
#     if fill_between_points > 0:
#         x = np.linspace(0, l - 1, l * (fill_between_points + 1))
#         x = np.interp(x, np.arange(l), line1["x"])
#         y1 = np.linspace(0, l - 1, l * (fill_between_points + 1))
#         y1 = np.interp(y1, np.arange(l), line1["y"])
#         y2 = np.linspace(0, l - 1, l * (fill_between_points + 1))
#         y2 = np.interp(y2, np.arange(l), line2["y"])
#     for index in range(len(x) - 1):
#         if y1[index + 1] > y2[index + 1]:
#             fill_color = color1
#         else:
#             fill_color = color2
#         ax.fill_between([x[index], x[index + 1]], [y1[index], y1[index + 1]],
#                         [y2[index], y2[index + 1]], color=fill_color, alpha=0.5,
#                         ec=None)

def plot_fillspace(ax: Axes, lines: List[Dict[str, Any]], fill_between_points: int = 9) -> None:
    X, Y, Color = [], [], []
    l = len(lines[0]["x"])
    for line in lines:
        x = line["x"]
        y = line["y"]
        if fill_between_points > 0:
            x = np.linspace(0, l - 1, l * (fill_between_points + 1))
            x = np.interp(x, np.arange(l), line["x"])
            y = np.linspace(0, l - 1, l * (fill_between_points + 1))
            y = np.interp(y, np.arange(l), line["y"])
        X.append(x)
        Y.append(y)
        Color.append(line["color"])
    X = np.array(X).T
    Y = np.array(Y).T
    for i in range(X.shape[0] - 1):
        max_idx = np.argmax(Y[i + 1, :])
        min_idx = np.argmin(Y[i + 1, :])
        ax.fill_between(X[i:i + 2, 0], Y[i:i + 2, max_idx], Y[i:i + 2, min_idx], color=Color[max_idx], alpha=0.5,
                        ec=None)


def show_background_image(ax: Axes, image_path: str) -> None:
    background_image = imageio.imread(image_path)
    im = background_image[:, :, :]
    ax.imshow(im)
