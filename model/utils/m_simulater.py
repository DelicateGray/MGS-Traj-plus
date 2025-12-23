import joblib
import numpy as np
from typing import Any
import time
import os
from tabulate import tabulate


class MonteCarloSimulater:
    def __init__(self, seq_length: int, mask_ratio: float, iteration: int, limitation: np.ndarray[Any, np.int_]):
        self.N = seq_length
        self.r = mask_ratio
        self.iter = iteration
        self.l_min, self.l_max = limitation

    def get_prob(self) -> np.ndarray:
        prob = []
        for i in range(self.iter):
            index = np.random.permutation(self.N)
            masked_idx = index[:int(self.r * self.N)]
            unmasked_idx = index[int(self.r * self.N):]
            count = 0
            for idx in masked_idx:
                dist = np.abs(unmasked_idx - idx)
                if self.l_min <= np.min(dist) <= self.l_max:
                    count += 1
            prob.append(count / masked_idx.shape[0])
        return np.array(prob)


if __name__ == '__main__':
    # 1431936 = 144640 * 11 * 0.9 < 1.5e6
    SimProb = {}
    for i in range(1, 10):
        ratio = i / 10
        mc_sim = MonteCarloSimulater(15, ratio, int(1.5e6), np.array([1, 5]))
        cur_prob = mc_sim.get_prob()
        SimProb.update({ratio: cur_prob})
        # print(" 掩码比例: ", ratio, " 概率: ", cur_prob.mean())
    SimProbTable = tabulate([["ratio"] + [f"{key:.1f}" for key in SimProb.keys()],
                             ["prob"] + [f"{value.mean():.6f}" for value in SimProb.values()]],
                            headers="firstrow",
                            tablefmt="simple_outline",
                            numalign="center",
                            stralign="center", )
    print(f"{SimProbTable}\n")
    log_dir = f"../../log/kinematic/{time.strftime('%Y-%m-%d-%H_%M_%S', time.localtime())}"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    with open(f"{log_dir}/sim_log.txt", "w") as file:
        file.write(f"\n{SimProbTable}")
    joblib.dump(SimProb, f"{log_dir}/sim_prob.pkl")
