import os
import torch
import hydra
import numpy as np
import uuid
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
from typing import Optional, List, Union, Any
from tqdm import tqdm
import logging
from copy import deepcopy
from contextlib import contextmanager

from verl import DataProto
from verl.utils import hf_tokenizer, hf_processor
from verl.utils.fs import copy_to_local
from verl.trainer.main_ppo import create_rl_dataset
from verl.utils.torch_functional import get_response_mask, pad_2d_list_to_length
from verl.utils.debug import GPUMemoryLogger
from verl.third_party.vllm import vllm_version

from vllm import LLM, SamplingParams
from vllm.distributed import parallel_state as vllm_ps
from vllm.lora.request import LoRARequest
from tensordict import TensorDict
import websocket
import json
import base64
import time
import threading

logger = logging.getLogger(__name__)

class WebSocketClient:
    """WebSocket客户端，用于与服务器通信"""
    
    def __init__(self, server_url: str, identity: str, retry_times: int = 5):
        self.server_url = server_url
        self.identity = identity
        self.retry_times = retry_times
        self.ws = None
        self.connected = False
        print(f"[Inference] Initializing WebSocket client to {server_url}")
        self.connect()
        
    def connect(self):
        """连接到WebSocket服务器"""
        for i in range(self.retry_times):
            try:
                print(f"[Inference] Attempting to connect to {self.server_url} (attempt {i+1}/{self.retry_times})")
                self.ws = websocket.WebSocket()
                self.ws.settimeout(10)  # 设置连接超时
                self.ws.connect(self.server_url)
                # 发送身份标识
                self.ws.send(self.identity)
                self.connected = True
                print(f"[Inference] Successfully connected to server at {self.server_url} as {self.identity}")
                return
            except Exception as e:
                print(f"[Inference] Failed to connect to server (attempt {i+1}): {e}")
                if i < self.retry_times - 1:
                    time.sleep(5)  # 等待5秒后重试
                else:
                    self.connected = False
                    raise ConnectionError(f"Failed to connect to server after {self.retry_times} attempts")
    
    def send_message(self, message: dict, timeout: int = 30) -> dict:
        """发送消息并等待响应"""
        if not self.connected:
            self.connect()
            
        try:
            self.ws.settimeout(timeout)
            self.ws.send(json.dumps(message))
            response = self.ws.recv()
            return json.loads(response)
        except Exception as e:
            print(f"[Inference] Error sending message: {e}")
            self.connected = False
            raise
    
    def upload_rollouts(self, file_path: str, batch_idx: int, dataset_name: str) -> bool:
        """上传rollouts文件到服务器"""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                with open(file_path, "rb") as f:
                    file_data = f.read()
                
                encoded_data = base64.b64encode(file_data).decode('utf-8')
                filename = os.path.basename(file_path)
                
                print(f"[Inference] Uploading {filename} (size: {len(file_data)} bytes) to server...")
                
                message = {
                    "type": "upload_rollouts",
                    "filename": filename,
                    "file_data": encoded_data,
                    "batch_idx": batch_idx,
                    "dataset_name": dataset_name
                }
                
                response = self.send_message(message, timeout=60)  # 增加超时时间
                
                if response.get("type") == "upload_success":
                    print(f"[Inference] Successfully uploaded {filename} to server")
                    print(f"[Inference] Server queue size: {response.get('queue_size', 'Unknown')}")
                    return True
                else:
                    print(f"[Inference] Failed to upload {filename}: {response}")
                    return False
                    
            except Exception as e:
                print(f"[Inference] Error uploading rollouts {file_path} (attempt {attempt+1}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(2)  # 短暂等待后重试
                    try:
                        self.connect()  # 尝试重新连接
                    except:
                        pass
                else:
                    return False
        
        return False
    
    def ping_server(self) -> bool:
        """测试服务器连接"""
        try:
            response = self.send_message({"type": "ping"}, timeout=5)
            return response.get("type") == "pong"
        except:
            return False
    
    def close(self):
        """关闭连接"""
        if self.ws:
            try:
                self.ws.close()
            except:
                pass
            self.connected = False

def compute_response_mask(data: DataProto):
    """Compute the attention mask for the response part of the sequence."""
    responses = data.batch["responses"]
    response_length = responses.size(1)
    attention_mask = data.batch["attention_mask"]
    return attention_mask[:, -response_length:]

def _pre_process_inputs(pad_token_id, prompt_token_ids: torch.Tensor) -> List[int]:
    """Remove the left padding in the prompt token_id"""
    non_pad_index = torch.nonzero(prompt_token_ids != pad_token_id, as_tuple=False)[0][0]
    token_ids = prompt_token_ids[non_pad_index:].tolist()
    return token_ids

def _repeat_interleave(value: Union[torch.Tensor, np.ndarray], repeats: int) -> Union[torch.Tensor, List[Any]]:
    if isinstance(value, torch.Tensor):
        return value.repeat_interleave(repeats, dim=0)
    else:
        return np.repeat(value, repeats, axis=0)


class VLLMInferenceEngine:
    """VLLM inference engine that directly uses vLLM LLM for generation"""
    
    def __init__(self, config: DictConfig):
        self.config = config
        self.setup_model_and_tokenizer()
        self.setup_vllm_engine()

        self.ws_client = None
        if self.config.inference.get("enable_server", True):
            server_ip = "10.0.2.111"
            server_port = "8765"
            server_url = f"ws://{server_ip}:{server_port}"
            
            print(f"[Inference] Initializing WebSocket client for server: {server_url}")
            print(f"[Inference] Running on inference node: 10.0.0.128")
            try:
                self.ws_client = WebSocketClient(server_url, "node_inference")
                
                if self.ws_client.ping_server():
                    print("[Inference] Server connection test successful")
                else:
                    print("[Inference] Server connection test failed")
            except Exception as e:
                print(f"[Inference] Failed to initialize WebSocket client: {e}")
                self.ws_client = None

    def setup_model_and_tokenizer(self):
        """Setup model and tokenizer"""
        local_path = copy_to_local(
            self.config.model.path, 
            use_shm=self.config.model.get('use_shm', False)
        )
        
        trust_remote_code = self.config.model.get("trust_remote_code", False)
        self.tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        self.processor = hf_processor(local_path, trust_remote_code=trust_remote_code)
        
        from transformers import AutoConfig
        self.model_hf_config = AutoConfig.from_pretrained(
            local_path, 
            trust_remote_code=trust_remote_code
        )
        
        self.local_model_path = local_path
        
    def setup_vllm_engine(self):
        """Setup the VLLM engine directly"""
        # 设置分布式环境变量（如果需要）
        if not torch.distributed.is_initialized():
            os.environ.setdefault('MASTER_ADDR', 'localhost')
            os.environ.setdefault('MASTER_PORT', '29500')
            os.environ.setdefault('RANK', '0')
            os.environ.setdefault('WORLD_SIZE', '1')
            os.environ.setdefault('LOCAL_RANK', '0')
            
            try:
                torch.distributed.init_process_group(
                    backend="nccl",
                    rank=0,
                    world_size=1
                )
            except:
                torch.distributed.init_process_group(
                    backend="gloo",
                    rank=0,
                    world_size=1
                )
        
        # 获取配置参数
        tensor_parallel_size = self.config.rollout.get("tensor_model_parallel_size", 1)
        max_num_batched_tokens = self.config.rollout.get("max_num_batched_tokens", 8192)
        
        # 检查模型最大长度
        rope_scaling_config = getattr(self.model_hf_config, "rope_scaling", None)
        if not rope_scaling_config:
            max_position_embeddings = None
            if hasattr(self.model_hf_config, "max_position_embeddings"):
                max_position_embeddings = self.model_hf_config.max_position_embeddings
            elif hasattr(self.model_hf_config, "llm_config") and hasattr(self.model_hf_config.llm_config, "max_position_embeddings"):
                max_position_embeddings = self.model_hf_config.llm_config.max_position_embeddings
            elif hasattr(self.model_hf_config, "text_config") and hasattr(self.model_hf_config.text_config, "max_position_embeddings"):
                max_position_embeddings = self.model_hf_config.text_config.max_position_embeddings
            
            if max_position_embeddings:
                assert max_position_embeddings >= self.config.rollout.prompt_length + self.config.rollout.response_length, \
                    "Model context length should be greater than total sequence length"

        max_model_len = int(self.config.rollout.max_model_len or 
                           self.config.rollout.prompt_length + self.config.rollout.response_length)

        trust_remote_code = self.config.model.get("trust_remote_code", False)
        load_format = "dummy" if self.config.rollout.load_format.startswith("dummy") else self.config.rollout.load_format

        # 初始化 vLLM 引擎
        self.inference_engine = LLM(
            model=self.local_model_path,
            enable_sleep_mode=True,
            tensor_parallel_size=tensor_parallel_size,
            distributed_executor_backend="external_launcher",
            dtype=self.config.rollout.dtype,
            enforce_eager=self.config.rollout.enforce_eager,
            gpu_memory_utilization=self.config.rollout.gpu_memory_utilization,
            disable_custom_all_reduce=True,
            disable_mm_preprocessor_cache=True,
            skip_tokenizer_init=False,
            max_model_len=max_model_len,
            load_format=load_format,
            disable_log_stats=self.config.rollout.disable_log_stats,
            max_num_batched_tokens=max_num_batched_tokens,
            enable_chunked_prefill=self.config.rollout.enable_chunked_prefill,
            enable_prefix_caching=True,
            trust_remote_code=trust_remote_code,
            seed=self.config.rollout.get("seed", 0),
        )
        
        # 设置采样参数
        kwargs = dict(
            n=1,
            logprobs=0,
            max_tokens=self.config.rollout.response_length,
        )
        
        if vllm_version != "0.3.1":
            kwargs["detokenize"] = False

        # 从配置中添加采样参数
        for k in self.config.rollout.keys():
            if hasattr(SamplingParams(), str(k)):
                kwargs[k] = self.config.rollout.get(k)

        self.sampling_params = SamplingParams(**kwargs)
        self.pad_token_id = self.tokenizer.pad_token_id
        
        logger.info(f"VLLM engine initialized with sampling params: {kwargs}")

    @contextmanager
    def update_sampling_params(self, **kwargs):
        """Update sampling params temporarily"""
        old_sampling_params_args = {}
        if kwargs:
            for key, value in kwargs.items():
                if hasattr(self.sampling_params, key):
                    old_value = getattr(self.sampling_params, key)
                    old_sampling_params_args[key] = old_value
                    setattr(self.sampling_params, key, value)
        yield
        # Roll back to previous sampling params
        for key, value in old_sampling_params_args.items():
            setattr(self.sampling_params, key, value)

    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
        """Generate sequences using vLLM engine - extracted from vllm_rollout_spmd.py"""
        
        # Rebuild vllm cache engine if needed
        if (vllm_version in ("0.5.4", "0.6.3") and 
            self.config.rollout.free_cache_engine):
            self.inference_engine.init_cache_engine()

        idx = prompts.batch["input_ids"]  # (bs, prompt_length)
        attention_mask = prompts.batch["attention_mask"]
        position_ids = prompts.batch["position_ids"]
        eos_token_id = prompts.meta_info["eos_token_id"]
        batch_size = idx.size(0)

        non_tensor_batch = prompts.non_tensor_batch
        if "raw_prompt_ids" not in non_tensor_batch:
            non_tensor_batch["raw_prompt_ids"] = np.array(
                [_pre_process_inputs(self.pad_token_id, idx[i]) for i in range(batch_size)], 
                dtype=object
            )

        if batch_size != len(non_tensor_batch["raw_prompt_ids"]):
            raise RuntimeError("vllm sharding manager is not work properly.")

        # Prepare vLLM inputs
        if "multi_modal_data" in non_tensor_batch:
            vllm_inputs = []
            for raw_prompt_ids, multi_modal_data in zip(
                non_tensor_batch.pop("raw_prompt_ids"), 
                non_tensor_batch.pop("multi_modal_data")
            ):
                vllm_inputs.append({
                    "prompt_token_ids": raw_prompt_ids, 
                    "multi_modal_data": multi_modal_data
                })
        else:
            vllm_inputs = [
                {"prompt_token_ids": raw_prompt_ids} 
                for raw_prompt_ids in non_tensor_batch.pop("raw_prompt_ids")
            ]

        # Ensure prompt_token_ids is list[int]
        for input_data in vllm_inputs:
            if isinstance(input_data["prompt_token_ids"], np.ndarray):
                input_data["prompt_token_ids"] = input_data["prompt_token_ids"].tolist()
            elif not isinstance(input_data["prompt_token_ids"], list):
                raise TypeError(f"prompt_token_ids must be a list or numpy array, got {type(input_data['prompt_token_ids'])}")
        
        # Set sampling parameters based on meta_info
        do_sample = prompts.meta_info.get("do_sample", True)
        is_validate = prompts.meta_info.get("validate", False)
        
        if not do_sample:
            sampling_kwargs = {
                "best_of": 1,
                "top_p": 1.0,
                "top_k": -1,
                "min_p": 0.0,
                "temperature": 0,
                "n": 1,
            }
        elif is_validate:
            sampling_kwargs = {
                "top_k": self.config.rollout.val_kwargs.top_k,
                "top_p": self.config.rollout.val_kwargs.top_p,
                "temperature": self.config.rollout.val_kwargs.temperature,
                "n": 1,
            }
        else:
            sampling_kwargs = {}

        # Generate sequences
        with self.update_sampling_params(**sampling_kwargs):
            outputs = self.inference_engine.generate(
                prompts=vllm_inputs,
                sampling_params=self.sampling_params,
                use_tqdm=False,
            )

            # Process outputs
            response = []
            for output in outputs:
                for sample_id in range(len(output.outputs)):
                    response.append(output.outputs[sample_id].token_ids)

            response = pad_2d_list_to_length(
                response, self.pad_token_id, 
                max_length=self.config.rollout.response_length
            ).to(idx.device)

            # Handle multiple responses per prompt
            if self.sampling_params.n > 1 and do_sample:
                idx = _repeat_interleave(idx, self.sampling_params.n)
                attention_mask = _repeat_interleave(attention_mask, self.sampling_params.n)
                position_ids = _repeat_interleave(position_ids, self.sampling_params.n)
                batch_size = batch_size * self.sampling_params.n

                if "multi_modal_inputs" in non_tensor_batch.keys():
                    non_tensor_batch["multi_modal_inputs"] = _repeat_interleave(
                        non_tensor_batch["multi_modal_inputs"], self.sampling_params.n
                    )
                if "tools_kwargs" in non_tensor_batch.keys():
                    non_tensor_batch["tools_kwargs"] = _repeat_interleave(
                        non_tensor_batch["tools_kwargs"], self.sampling_params.n
                    )

            seq = torch.cat([idx, response], dim=-1)

        # Update position_ids and attention_mask
        response_length = response.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.unsqueeze(0).expand(batch_size, -1)
        
        if position_ids.dim() == 3:  # qwen2vl mrope
            delta_position_id = delta_position_id.view(batch_size, 1, -1).expand(batch_size, 3, -1)

        response_position_ids = position_ids[..., -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
        
        response_attention_mask = get_response_mask(
            response_id=response, eos_token=eos_token_id, dtype=attention_mask.dtype
        )
        attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)

        # Create output batch
        batch = TensorDict(
            {
                "prompts": idx,
                "responses": response,
                "input_ids": seq,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            },
            batch_size=batch_size,
        )

        # Free vllm cache engine if needed
        if (vllm_version in ("0.5.4", "0.6.3") and 
            self.config.rollout.free_cache_engine):
            self.inference_engine.free_cache_engine()

        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch)

    def create_datasets(self):
        """Create training and validation datasets"""
        self.train_dataset = create_rl_dataset(
            self.config.data.train_files,
            self.config.data,
            self.tokenizer,
            self.processor
        )
        
        self.val_dataset = create_rl_dataset(
            self.config.data.val_files,
            self.config.data,
            self.tokenizer,
            self.processor
        )
        
        logger.info(f"Train dataset size: {len(self.train_dataset)}")
        logger.info(f"Validation dataset size: {len(self.val_dataset)}")
        
    def create_dataloader(self, dataset, batch_size: int, shuffle: bool = False):
        """Create a DataLoader for the dataset"""
        from torch.utils.data import DataLoader
        from verl.utils.dataset.rl_dataset import collate_fn
        
        return DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=1,
            collate_fn=collate_fn,
            drop_last=False
        )
        
    def generate_rollouts(self, dataset, dataset_name: str):
        """Generate rollouts for a given dataset and save each batch separately"""
        dataloader = self.create_dataloader(
            dataset, 
            batch_size=self.config.data.train_batch_size,
            shuffle=False
        )
        
        output_dir = Path(self.config.inference.output_dir) / dataset_name
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Starting to generate {dataset_name} rollouts")
        saved_files = []
                
        for batch_idx, batch_dict in enumerate(tqdm(dataloader, desc=f"Generating {dataset_name} rollouts")):
            try:
                # Convert to DataProto format
                batch: DataProto = DataProto.from_single_dict(batch_dict)

                # Pop keys following ray_trainer approach
                batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
                non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]
                if "multi_modal_data" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("multi_modal_data")
                if "raw_prompt" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("raw_prompt")
                if "tools_kwargs" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("tools_kwargs")
                
                gen_batch = batch.pop(
                    batch_keys=batch_keys_to_pop,
                    non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
                )

                # Set meta_info
                gen_batch.meta_info = {
                    "eos_token_id": self.tokenizer.eos_token_id,
                    "pad_token_id": self.tokenizer.pad_token_id,
                    "recompute_log_prob": False,
                    "do_sample": self.config.rollout.do_sample,
                    "validate": False,
                }
                
                # Generate sequences using our extracted method
                gen_batch_output = self.generate_sequences(gen_batch)
                
                # Add UUID identifiers
                batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
                )
                
                # Repeat to align with repeated responses in rollout
                batch = batch.repeat(repeat_times=self.config.rollout.n, interleave=True)
                
                # Union generated data
                batch = batch.union(gen_batch_output)
                        
                # Save current batch
                output_file = output_dir / f"rollouts_{batch_idx}.pt"
                torch.save(batch, output_file)
                saved_files.append(output_file)
                
                logger.info(f"Batch {batch_idx} saved to: {output_file}")
                logger.info(f"Batch {batch_idx} contains {batch.batch.batch_size[0]} samples")
                
                if self.ws_client:
                    success = self.ws_client.upload_rollouts(
                        file_path=str(output_file),
                        batch_idx=batch_idx,
                        dataset_name=dataset_name
                    )
                    if success:
                        print(f"[Inference] Successfully uploaded batch {batch_idx} to server (10.0.2.111)")
                    else:
                        print(f"[Inference] Failed to upload batch {batch_idx} to server")
                
                if (batch_idx + 1) % 10 == 0:
                    print(f"[Inference] {batch_idx + 1}/{len(dataloader)} batches processed")
                    
            except Exception as e:
                logger.error(f"Error processing batch {batch_idx}: {e}")
                import traceback
                logger.error(f"Full traceback: {traceback.format_exc()}")
                continue

        logger.info(f"Generation completed for {dataset_name}")
        logger.info(f"Total {len(saved_files)} batch files saved to: {output_dir}")
        
        return saved_files
    
    def run_inference(self):
        """Run inference pipeline"""
        logger.info("Starting VLLM inference pipeline...")
        
        try:
            self.create_datasets()
            train_files = self.generate_rollouts(self.train_dataset, "train")
            val_files = self.generate_rollouts(self.val_dataset, "val")
            
            logger.info("Inference completed!")
            if self.ws_client:
                logger.info("Keeping WebSocket connection alive for future uploads...")
                logger.info("Press Ctrl+C to shutdown...")
                try:
                    while True:
                        time.sleep(1)
                except KeyboardInterrupt:
                    logger.info("Shutting down inference engine...")
                    self.ws_client.close()
            
            return {
                "train_files": train_files,
                "val_files": val_files
            }
        except Exception as e:
            if self.ws_client:
                self.ws_client.close()
            raise

@hydra.main(config_path="config", config_name="vllm_inferencer", version_base=None)
def main(config: DictConfig):
    """Main function"""
    print("=== VLLM Inference Configuration ===")
    print(OmegaConf.to_yaml(config))
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create inference engine
    inference_engine = VLLMInferenceEngine(config)
    
    # Run inference
    results = inference_engine.run_inference()
    
    print("Inference task completed!")
    print(f"Results saved in: {config.inference.output_dir}")
    print(f"Train files: {len(results['train_files'])}")
    print(f"Validation files: {len(results['val_files'])}")


if __name__ == "__main__":
    main()