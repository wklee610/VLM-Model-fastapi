import os
import json
import torch
from typing import Union
from vllm import (
    LLM, 
    SamplingParams
)
from transformers import (
    AutoProcessor, 
    LlavaOnevisionForConditionalGeneration
)

class ModelContainer:
    def __init__(
        self, 
        model: Union[LlavaOnevisionForConditionalGeneration, LLM],
        processor: AutoProcessor, 
        device: torch.device,
        torch_dtype: torch.dtype,
        tensor_type: str,
        max_new_tokens: int,
        system_prompt: str,
        few_shots: json,
        sampling_params: Union[SamplingParams, None]
    ) -> None:
        self.model = model
        self.processor = processor
        self.device = device
        self.torch_dtype = torch_dtype
        self.tensor_type = tensor_type
        self.max_new_tokens = max_new_tokens
        self.system_prompt = system_prompt
        self.few_shots = few_shots
        self.sampling_params = sampling_params


class ModelManager:
    def __init__(
        self,
        env: object,
        device: torch.device,
        tensor_parallel_size: int
    ) -> None:
        self.use_vllm = env.USE_VLLM
        self.model_path = env.MODEL_PATH
        self.torch_dtype = env.TORCH_DTYPE
        self.attn_implementation = env.ATTN_IMPLEMENTATION
        self.device_map = env.DEVICE_MAP
        self.device = device
        self.tensor_type = env.TENSOR_TYPE
        self.max_new_tokens = env.MAX_NEW_TOKENS
        self.system_prompt = env.SYSTEM_PROMPT
        self.few_shots = env.FEW_SHOTS
        self.tensor_parallel_size = tensor_parallel_size
        self.max_model_len = env.MAX_MODEL_LEN
        self.max_num_batched_tokens = env.MAX_NUM_BATCHED_TOKENS
        self.temperature = env.TEMPERATURE
        self.top_p = env.TOP_P
        
    def load(
        self,
    ) -> None:
        if self.use_vllm:
            self.model = LLM(
                model=self.model_path,
                max_model_len=self.max_model_len,
                max_num_batched_tokens=self.max_num_batched_tokens,
                dtype=str(self.torch_dtype).replace("torch.", ""),
                tensor_parallel_size=self.tensor_parallel_size,
            )
            # os.environ["TOKENIZERS_PARALLELISM"] = "false"
        else:
            self.model = LlavaOnevisionForConditionalGeneration.from_pretrained(
                self.model_path,
                torch_dtype=self.torch_dtype,
                attn_implementation=self.attn_implementation,
                device_map=self.device_map,
            )
        self.processor = AutoProcessor.from_pretrained(self.model_path, use_fast=True)

    def get_sampling_params(
            self, 
            use_vllm: bool
        ) -> Union[SamplingParams, None]:
        if use_vllm:
            return SamplingParams(
                max_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p
            )
        return None

    def get_model(
        self,
    ) -> ModelContainer:
        return ModelContainer(
            self.model, 
            self.processor, 
            self.device,
            self.torch_dtype,
            self.tensor_type,
            self.max_new_tokens,
            self.system_prompt,
            self.few_shots,
            self.get_sampling_params(self.use_vllm)
        )
