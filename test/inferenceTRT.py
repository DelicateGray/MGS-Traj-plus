import os
import torch
import numpy as np
import tensorrt as trt
from config import load_config
from calibrator import PredModelCalibrator
import common

os.environ['TRT_LAZY_LOAD'] = '1'  # 启用Nvidia lazy load，好像是cuda12+引入的新特性，没啥用但会报警告
TRT_LOGGER = trt.Logger(trt.Logger.ERROR)  # 建立TensortRT日志，全局可用，TensortRT优化和部署的关键依赖项，会记录详细的优化和部署信息

class Inference:
    def __init__(self, model_path: str = None, engine_path: str = None, accuracy="float32", sample_ratio: float = 1.0):
        self.model_path = model_path
        self.engine_path = engine_path
        self.accuracy = accuracy
        self.sample_ratio = sample_ratio
        self.config = load_config("../config.yaml")
        self.B, self.N, self.Th, self.Tf, self.D = self.config['common']['B'], self.config['common']['N'], \
            self.config['common']['Th'], self.config['common']['Tf'], self.config['common']["D"]
        self.data_dir = self.config['common']['data_dir']

    # ✅ 新增：导出结构化剪枝后的模型为 ONNX
    def export_pruned_model_to_onnx(self, pruned_model: torch.nn.Module, onnx_path: str):
        dummy_traj = torch.randn((self.B, self.N, self.Th, self.D), dtype=torch.float32)
        dummy_mask = torch.zeros((self.B, self.N, self.Th), dtype=torch.bool)
        dummy_mask[:, 6:, :] = 1
        pruned_model.eval()
        torch.onnx.export(
            pruned_model,
            (dummy_traj, dummy_mask),
            onnx_path,
            input_names=["traj", "traj_mask"],
            output_names=["pred"],
            opset_version=11
        )
        print(f"[ONNX] Model exported to: {onnx_path}")

    # ✅ 修改增强：支持 FP16 / INT8 / 稀疏 推理
    def build_trt_engine(self, onnx_path: str, engine_path: str, accuracy: str = "float32"):
        builder = trt.Builder(TRT_LOGGER)
        network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        network = builder.create_network(network_flags)
        config = builder.create_builder_config()
        parser = trt.OnnxParser(network, TRT_LOGGER)
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, common.GiB(5))

        if accuracy == "float32" and builder.platform_has_tf32:
            config.set_flag(trt.BuilderFlag.TF32)
        elif accuracy == "float16" and builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
        elif accuracy == "int8" and builder.platform_has_fast_int8:
            config.set_flag(trt.BuilderFlag.INT8)
            calibrator = PredModelCalibrator(self.B, 11, f"{self.data_dir}/pred/test_loader.pkl", "calib.cache")
            config.set_quantization_flag(trt.QuantizationFlag.CALIBRATE_BEFORE_FUSION)
            config.int8_calibrator = calibrator

        if builder.platform_has_sparse_weights:
            config.set_flag(trt.BuilderFlag.SPARSE_WEIGHTS)

        # ✅ 添加动态 profile 支持
        profile = builder.create_optimization_profile()
        traj_shape = (self.B, self.N, self.Th, self.D)
        mask_shape = (self.B, self.N, self.Th)
        profile.set_shape("traj", traj_shape, traj_shape, traj_shape)
        profile.set_shape("traj_mask", mask_shape, mask_shape, mask_shape)
        config.add_optimization_profile(profile)

        with open(onnx_path, 'rb') as model_file:
            if not parser.parse(model_file.read()):
                for error in range(parser.num_errors):
                    print(parser.get_error(error))

        engine_bytes = builder.build_serialized_network(network, config)
        with open(engine_path, 'wb') as f:
            f.write(engine_bytes)
        print(f"[TensorRT] Engine saved to: {engine_path}")

    def load_trt_engine(self, engine_path: str) -> Union[trt.ICudaEngine, None]:
        """
            加载序列化的TensorRT推理引引擎并反序列化
            参数
                onnx_path: ONNX模型路径
            返回值
                engine：反序列化的引擎
        """
        # 读取保存的引擎文件
        with open(engine_path, 'rb') as f:
            engine_bytes = f.read()
        # 反序列化为 TensorRT 引擎
        runtime = trt.Runtime(TRT_LOGGER)
        engine = runtime.deserialize_cuda_engine(engine_bytes)
        if engine is None:
            print("ERROR: Failed to load the engine.")
            return None
        return engine

    def infer_step_on_cpu(self, model_path: str, traj: torch.Tensor, traj_mask: torch.Tensor) -> tuple[
        torch.Tensor, float]:
        """
            在CPU上执行一个批量数据的推理，并对结果进行后处理
            参数
                model_path: pytorch模型的路径
                traj: 一个批量的轨迹数据
                traj_mask: 一个批量的轨迹掩码数据
            返回值
                pred：预测轨迹
                avg_inference_time：单个样本的平均推理时间，单位毫秒
        """
        model = torch.load(model_path).to(torch.device("cpu"))
        model.eval()
        B = traj.shape[0]
        with torch.no_grad():
            t1 = time.time()
            pred = model(traj.to(torch.device("cpu")), traj_mask.to(torch.device("cpu")))
            t2 = time.time()
            avg_inference_time = (t2 - t1) * 1000 / B
        return pred, avg_inference_time

    def infer_step_on_gpu(self, engine_path: str, traj: np.ndarray, traj_mask: np.ndarray) -> tuple[np.ndarray, float]:
        """
            在GPU上执行一个批量数据的推理，并对结果进行后处理
            参数
                engine_path: TensorRT推理引擎的路径
                traj: 一个批量的轨迹数据
                traj_mask: 一个批量的轨迹掩码数据
            返回值
                pred：预测轨迹
                avg_inference_time：单个样本的平均推理时间，单位毫秒
        """
        B = traj.shape[0]
        engine = self.load_trt_engine(engine_path)
        inputs, outputs, bindings, stream = common.allocate_buffers(engine, B)
        context = engine.create_execution_context()
        np.copyto(inputs[0].host, traj.ravel())
        np.copyto(inputs[1].host, traj_mask.ravel())
        out, inference_time = common.do_inference(
            context,
            engine=engine,
            bindings=bindings,
            inputs=inputs,
            outputs=outputs,
            stream=stream,
        )
        pred = out[0].reshape(B, self.Tf, 2)
        avg_inference_time = inference_time / B
        return pred, avg_inference_time

    def load_test_data(self, data_path: str):
        """
            加载轨迹预测测试集数据
            参数
                data_path: 测试集数据路径
            返回值
                test_data：测试集轨迹和掩码的dataloader对象
                test_meta：测试集元数据的多级列表
        """
        test_data = joblib.load(f"{data_path}/test_loader.pkl")
        test_meta = joblib.load(f"{data_path}/test_meta.pkl")

        return test_data, test_meta

    def excute(self, run_on_cpu: bool, run_on_gpu: bool):
        """
            执行优化、推理和后处理，并保存相关指标和数据
            参数
                run_on_cpu: 是否在CPU上执行推理
                run_on_gpu: 是否在GPU上执行推理
        """
        test_data, test_meta = self.load_test_data(f"{self.data_dir}/pred")
        test_bacth_num = len(test_data)
        test_bar = tqdm(range(test_bacth_num))
        # 推理的RMSE和推理时间指标
        InferMetrics = {"RMSE_GPU": RMSE(), "RMSE_CPU": RMSE(), "T_GPU": InferenceTime(), "T_CPU": InferenceTime()}
        # 最大推理时间指标
        MaxTime = {"T_GPU": [], "T_CPU": []}
        if self.engine_path is None and run_on_gpu:
            base_name = f"{self.log_dir}/pred_model_{self.accuracy}"
            self.engine_path = f"{base_name}.engine"
            self.export_onnx(self.model_path, f"{base_name}.onnx")
            self.build_trt_engine(f"{base_name}.onnx", self.engine_path, self.accuracy)
        for i, (traj_batch, traj_mask_batch), meta in zip(test_bar, test_data, test_meta):
            if i < int(test_bacth_num * self.sample_ratio):
                traj = traj_batch[:, :, :self.Th, :]
                traj_mask = traj_mask_batch[:, :, :self.Th]
                target = traj_batch[:, 0, self.Th:, :2]
                target_m = CoordinateInverse(
                    [target], meta, torch.arange(len(meta))).batch_inverse()[0]
                t_cpu, t_gpu, pred_gpu = 0.0, 0.0, np.array([])
                if run_on_cpu:
                    pred_cpu, t_cpu = self.infer_step_on_cpu(
                        self.model_path,
                        traj,
                        traj_mask)
                    pred_cpu_m = CoordinateInverse(
                        [pred_cpu], meta, torch.arange(len(meta))).batch_inverse()[0]
                    InferMetrics["RMSE_CPU"].update(pred_cpu_m, target_m)
                    InferMetrics["T_CPU"].update(torch.tensor(t_cpu))
                    MaxTime["T_CPU"].append(t_cpu)
                if run_on_gpu:
                    pred_gpu, t_gpu = self.infer_step_on_gpu(self.engine_path, traj.numpy().astype(np.float32),
                                                             traj_mask.numpy().astype(np.bool_))
                    pred_gpu_m = CoordinateInverse(
                        [pred_gpu], meta, torch.arange(len(meta))).batch_inverse()[0]
                    InferMetrics["RMSE_GPU"].update(torch.tensor(pred_gpu_m), target_m)
                    InferMetrics["T_GPU"].update(torch.tensor(t_gpu))
                    MaxTime["T_GPU"].append(t_gpu)
                test_bar.set_description(
                    f"<Infering>  [T_CPU(ms)]: {t_cpu:.5f}  [T_GPU(ms)]: {t_gpu:.5f}"
                )
            else:
                break
        for key in InferMetrics.keys():
            InferMetrics[key] = InferMetrics[key].compute()
        for key in MaxTime.keys():
            MaxTime[key] = np.array(MaxTime[key]).max()
        TestMetricTable = tabulate(
            [["T_CPU(ms)", "T_GPU(ms)", "RMSE_CPU(m)", "RMSE_GPU(m)"],
             [f"{InferMetrics['T_CPU']:.5f}", f"{InferMetrics['T_GPU']:.5f}", f"{InferMetrics['RMSE_CPU']:.2f}",
              f"{InferMetrics['RMSE_GPU']:.2f}"]],
            headers="firstrow",
            tablefmt="simple_outline",
            numalign="center",
            stralign="center")
        print(f"MaxTime: {MaxTime}")
        print(f"{TestMetricTable}\n")
        with open(f"{self.log_dir}/inference_log.txt", "w") as file:
            file.write(TestMetricTable)
            file.write(f"MaxTime: {MaxTime}")


if __name__ == '__main__':
    inference = Inference(
        model_path="../log/pred/11/2024-12-22-15_39_37_k_dwa/checkpoint/checkpoint_49_0.004433_0.004190_.pth",
        engine_path=None,
        accuracy="float32",
        sample_ratio=1.0)
    inference.excute(run_on_cpu=True, run_on_gpu=True)
