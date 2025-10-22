from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import torch
from omegaconf import DictConfig
from torch.nn.utils.rnn import pad_sequence
from tensordict import TensorDict

from verl.utils.torch_functional import pad_sequence_to_length, get_response_mask

import asyncio
import websockets
import json
import os
from verl.workers.rollout.base import BaseRollout
from verl import DataProto
import torch.distributed as dist

if TYPE_CHECKING:
    from torch import nn

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

class PetalsRollout(BaseRollout):
    def __init__(self, config, module, tokenizer):
        super().__init__()
        self.config = config
        self.module = module
        self.tokenizer = tokenizer
        self.uri = config.get("petals_ws_uri", "ws://10.0.2.111:8765")

        self.http_server_dir = config.get("http_server_dir", "/opt/projects/test/model")

        # self.model_filename = config.get("petals_model_filename", "model.safetensors")
        # self.model_url = config.get("petals_model_url", f"http://10.0.2.111:8000/{self.model_filename}")
        # self.model_path = os.path.join(self.http_server_dir, self.model_filename)
        
        self.prompts_filename = config.get("petals_prompts_filename", "prompts.json")
        self.prompts_url = config.get("petals_prompts_url", f"http://10.0.2.111:8000/{self.prompts_filename}")
        self.prompts_path = os.path.join(self.http_server_dir, self.prompts_filename)

        self.n = config.get("n", 1)


        for var in ["http_proxy", "https_proxy", "all_proxy", "socks_proxy",
                "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "SOCKS_PROXY"]:
            os.environ.pop(var, None)

    def _save_current_model(self):
        from safetensors.torch import save_model
        if self.module is not None:

            state_dict = self.module.state_dict()
            save_model(self.module, self.model_path)
            logger.info(f"Model saved to {self.model_path}")
            return True
        return False

    def _save_prompts_to_file(self, prompts_data) -> bool:
        try:
            serializable_data = {"prompts": [], "n": self.n}
            tokenizer = self.tokenizer

            if hasattr(prompts_data, 'batch'):
                input_ids = prompts_data.batch["input_ids"].cpu().numpy()
                pad_token_id = prompts_data.meta_info.get("pad_token_id", tokenizer.pad_token_id)
                for i in range(input_ids.shape[0]):
                    tokens = input_ids[i]
                    if pad_token_id is not None:
                        non_pad_indices = tokens != pad_token_id
                        if non_pad_indices.any():
                            tokens = tokens[non_pad_indices]
                    prompt_text = tokenizer.decode(tokens.tolist(), skip_special_tokens=True)
                    serializable_data["prompts"].append(prompt_text)                    
            with open(self.prompts_path, 'w', encoding='utf-8') as f:
                json.dump(serializable_data, f)
            return True
            
        except Exception as e:
            print(f"保存提示数据到文件时出错: {e}")
            return False

    def _get_rollouts(self, prompts: DataProto, n: int) -> dict:
        if not self._save_prompts_to_file(prompts):
            logger.error("Failed to save prompts file")
            return {}

        import websocket

        ws = websocket.create_connection(self.uri)
        ws.send("node_verl")
        
        ws.send(json.dumps({
            "type": "prompts", 
            "filename": self.prompts_filename,
            "url": self.prompts_url,
            "n": n,
            }
        ))

        while True:
            msg = ws.recv()
            data = json.loads(msg)
            if data.get("type") == "new_rollouts":
                rollouts_url = data.get("url")
                if rollouts_url:
                    try:
                        import requests
                        response = requests.get(rollouts_url)
                        response.raise_for_status()
                        ws.close()
                        return response.json()
                    except Exception as e:
                        print(f"无法从URL获取rollouts数据: {e}")
                        ws.close()
                        return {}
                else:
                    logger.error("服务器返回的消息中没有URL")
                    ws.close()
                    return {}

    def generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
        # print(f"Generating sequences with prompts: {prompts}")
        input_ids = prompts.batch["input_ids"]
        attention_mask = prompts.batch["attention_mask"]
        position_ids = prompts.batch["position_ids"]
        eos_token_id = self.tokenizer.eos_token_id
        pad_token_id = self.tokenizer.pad_token_id
        if eos_token_id is None:
            eos_token_id = 0
        if pad_token_id is None:
            pad_token_id = 0

        is_validate = prompts.meta_info.get("validate", False)
        n = 1 if is_validate else self.n
        
        batch_size = input_ids.size(0)

        is_distributed = dist.is_initialized() if hasattr(dist, "is_initialized") else False
        is_main_process = not is_distributed or dist.get_rank() == 0
        device = input_ids.device

        if is_main_process:
            rollouts = self._get_rollouts(prompts, n)
            # print(f"Received rollouts on {device}: {rollouts}")
        else:
            rollouts = {}
            
        if is_distributed:
            if is_main_process:
                rollouts_str = json.dumps(rollouts)
                rollouts_bytes = rollouts_str.encode('utf-8')
                size_tensor = torch.tensor([len(rollouts_bytes)], dtype=torch.long, device=device)
            else:
                size_tensor = torch.tensor([0], dtype=torch.long, device=device)
                
            dist.broadcast(size_tensor, src=0)
            if not is_main_process:
                rollouts_bytes = bytearray(size_tensor.item())
                
            if is_main_process:
                bytes_tensor = torch.tensor(list(rollouts_bytes), dtype=torch.uint8, device=device)
            else:
                bytes_tensor = torch.zeros(size_tensor.item(), dtype=torch.uint8, device=device)
                
            dist.broadcast(bytes_tensor, src=0)
            if not is_main_process:
                rollouts_bytes = bytes_tensor.cpu().numpy().tobytes()
                rollouts_str = rollouts_bytes.decode('utf-8')
                rollouts = json.loads(rollouts_str)

        if isinstance(rollouts, str):
            rollouts = json.loads(rollouts)
        
        responses_token_ids = []

        for i, response in enumerate(rollouts): 
            responses_token_ids.append(response)
        
        max_response_len = max(len(tokens) for tokens in responses_token_ids)
        response_length = max(self.config.response_length, max_response_len)

        padded_responses = []
        for tokens in responses_token_ids:
            if len(tokens) > response_length:
                tokens = tokens[:response_length]
            padded = tokens + [pad_token_id] * (response_length - len(tokens))
            padded_responses.append(padded)

        if n > 1:
            idx = input_ids.repeat_interleave(n, dim=0)
            attention_mask = attention_mask.repeat_interleave(n, dim=0)
            position_ids = position_ids.repeat_interleave(n, dim=0)
            batch_size = batch_size * n
        else:
            idx = input_ids
            batch_size = batch_size

        response_tensor = torch.tensor(padded_responses, dtype=torch.long, device=idx.device)
        response_tensor = response_tensor.squeeze(1)
        seq = torch.cat([idx, response_tensor], dim=1)

        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.unsqueeze(0).repeat(batch_size, 1)
        response_position_ids = position_ids[:, -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=1)
        
        response_attention_mask = get_response_mask(response_id=response_tensor, 
                                                eos_token=eos_token_id, 
                                                dtype=attention_mask.dtype)
        attention_mask = torch.cat((attention_mask, response_attention_mask), dim=1)
        batch = TensorDict(
            {
                "prompts": idx,
                "responses": response_tensor,
                "input_ids": seq, 
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            },
            batch_size=batch_size,
        )
        
        return DataProto(batch=batch)