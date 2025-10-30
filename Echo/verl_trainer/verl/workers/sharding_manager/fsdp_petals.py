from verl.workers.sharding_manager.base import BaseShardingManager
from verl import DataProto

class FSDPPetalsShardingManager(BaseShardingManager):
    def __init__(self, module=None, inference_engine=None, model_config=None, 
                 full_params=False, device_mesh=None, offload_param=False):
        super().__init__()
        self.module = module
        self.inference_engine = inference_engine
        self.model_config = model_config
        self.full_params = full_params
        self.device_mesh = device_mesh
        self.offload_param = offload_param
        
    def preprocess_data(self, data: DataProto) -> DataProto:
        """预处理输入数据"""
        return data

    def postprocess_data(self, data: DataProto) -> DataProto:
        """后处理输出数据"""
        return data

    def __enter__(self):
        """上下文管理器入口"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        pass