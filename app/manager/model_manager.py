import torch
from transformers import (
    AutoProcessor, 
    LlavaOnevisionForConditionalGeneration
)

class ModelContainer:
    def __init__(
        self, 
        model: LlavaOnevisionForConditionalGeneration,
        processor: AutoProcessor, 
        device: torch.device,
        torch_dtype: torch.dtype,
        tensor_type: str,
        max_new_tokens: int
    ) -> None:
        self.model = model
        self.processor = processor
        self.device = device
        self.torch_dtype = torch_dtype
        self.tensor_type = tensor_type
        self.max_new_tokens = max_new_tokens


class ModelManager:
    def __init__(
        self,
        env: object,
        device: torch.device
    ) -> None:
        self.model_path = env.MODEL_PATH
        self.torch_dtype = env.TORCH_DTYPE
        self.attn_implementation = env.ATTN_IMPLEMENTATION
        self.device_map = env.DEVICE_MAP
        self.device = device
        self.tensor_type = env.TENSOR_TYPE
        self.max_new_tokens = env.MAX_NEW_TOKENS

    def load(
        self,
    ) -> None:
        self.model = LlavaOnevisionForConditionalGeneration.from_pretrained(
            self.model_path,
            torch_dtype=self.torch_dtype,
            attn_implementation=self.attn_implementation,
            device_map=self.device_map,
        )
        self.processor = AutoProcessor.from_pretrained(self.model_path, use_fast=True)

    def get_llava_model(
        self,
    ) -> ModelContainer:
        return ModelContainer(
            self.model, 
            self.processor, 
            self.device,
            self.torch_dtype,
            self.tensor_type,
            self.max_new_tokens
        )


    



