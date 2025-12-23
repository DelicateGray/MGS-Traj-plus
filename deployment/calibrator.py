import os
import random
import joblib
import numpy as np
import pycuda.driver as cuda
import tensorrt as trt
from config import *


class PredModelCalibrator(trt.IInt8MinMaxCalibrator):
    """
    预测模型的INT8量化校准类
    参数
        batch_size：批量大小
        bacth_num：校准批量数
        data_path：校准数据路径
        cache_file：校准缓存路径
    """

    def __init__(self, batch_size, bacth_num, data_path, cache_file):
        trt.IInt8MinMaxCalibrator.__init__(self)
        self.batch_size = batch_size
        self.batch_num = bacth_num
        self.data_path = data_path
        self.data = self.get_random_batches(self.data_path, bacth_num)
        self.cache_file = cache_file
        self.current_index = 0
        self.config = load_config("../config.yaml")
        self.N, self.Th, self.D = (
            self.config["common"]["N"],
            self.config["common"]["Th"],
            self.config["common"]["D"],
        )
        self.device_inputs = [
            cuda.mem_alloc(
                self.batch_size * self.N * self.Th * self.D * trt.float32.itemsize
            )
            for _ in range(2)
        ]

    def get_batch(self, names, p_str=None):
        """
        获取一个批量的校准数据，并存入显存缓冲区
        参数
            names：输入变量名列表
            p_str：没啥用的参数，默认为None
        返回值
            self.device_inputs：包含所有输入数据缓冲区指针的列表
        """
        if self.current_index >= self.batch_num:
            return None
        batch_data = self.data[self.current_index]
        traj = batch_data[0][:, :, : self.Th, :].numpy().astype(np.float32)
        traj_mask = batch_data[0][:, :, : self.Th].numpy().astype(np.bool_)
        cuda.memcpy_htod(self.device_inputs[0], traj.ravel())
        cuda.memcpy_htod(self.device_inputs[1], traj_mask.ravel())
        self.current_index += 1
        return self.device_inputs

    def read_calibration_cache(self):
        """
        读取校准缓存
        返回值
            校准缓存数据，如果存在返回，不存在返回 None
        """
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                return f.read()
        return None

    def write_calibration_cache(self, cache):
        """
        写入校准缓存
        参数
            cache：需要写入的校准缓存数据
        """
        with open(self.cache_file, "wb") as f:
            f.write(cache)
            f.flush()
            os.fsync(f)

    def get_batch_size(self):
        """
        获取批量大小
        返回值
            self.batch_size：批量大小
        """
        return self.batch_size

    def get_random_batches(self, data_path: str, batch_num: int):
        """
        随机获取 batch_num 个批量的校准数据
        参数
            data_path：校准数据路径
            batch_num：批量数量
        返回值
            batches：包含 batch_num 个随机批量的校准数据
        """
        test_data = joblib.load(data_path)
        batches = random.sample(
            [next(iter(test_data)) for _ in range(len(test_data))], batch_num
        )

        return batches
