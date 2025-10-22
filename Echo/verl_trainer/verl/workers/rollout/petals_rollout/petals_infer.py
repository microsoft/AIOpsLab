import torch
from transformers import AutoTokenizer
from petals import AutoDistributedModelForCausalLM

class PetalsInferencer:
    def __init__(
        self,
        model_name: str,
        initial_peers=None,
        use_fast_tokenizer=True,
        device=None
    ):
        self.model_name = model_name
        self.initial_peers = initial_peers or []
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=use_fast_tokenizer)
        self.model = AutoDistributedModelForCausalLM.from_pretrained(
            model_name,
            initial_peers=self.initial_peers
            # initial_peers=[
            #     "/ip4/10.0.0.128/tcp/44543/p2p/12D3KooWSMNs2yBwbWVLVhHsrRCHJzsp4RRKZH2BcPVKmhpt1oRF",
            #     "/ip4/127.0.0.1/tcp/44543/p2p/12D3KooWSMNs2yBwbWVLVhHsrRCHJzsp4RRKZH2BcPVKmhpt1oRF",
            # ]
        )
        if device is not None:
            self.model = self.model.to(device)

    def infer(
        self,
        prompt: str,
        max_new_tokens: int = 1024,
        do_sample: bool = True,
        temperature: float = 0.7,
        **generate_kwargs
    ):
        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt")
        attention_mask = inputs["attention_mask"]

        outputs = self.model.generate(
            inputs["input_ids"],
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.eos_token_id,
            do_sample=do_sample,
            temperature=temperature,
            **generate_kwargs
        )
        # print(f"outputs: {outputs}")
        # decoded_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # print(f"decoded_text: {decoded_text[:200]}...")
        return outputs

    def load_weights(self, weights_path: str):
        """
        加载safetensors权重到当前模型。
        仅适用于权重结构与当前模型完全一致的情况。
        """
        from safetensors.torch import load_file

        # 加载safetensors权重为state_dict
        state_dict = load_file(weights_path)
        # 加载权重到模型
        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"警告: 有缺失的权重: {missing}")
        if unexpected:
            print(f"警告: 有未使用的权重: {unexpected}")
        print(f"权重 {weights_path} 已加载到模型 {self.model_name}")
