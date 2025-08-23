import os
import torch
from dotenv import load_dotenv

load_dotenv(".env.local")

MODEL_NAME = os.getenv("MODEL_NAME")
PROJECT_TITLE = os.getenv("PROJECT_TITLE") or f"{MODEL_NAME}-fastapi"
MODEL_PATH = os.getenv("MODEL_PATH")
TORCH_DTYPE = os.getenv("TORCH_DTYPE")
TORCH_DTYPE = {
    "float16": torch.float16,
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "int8": torch.int8
}[TORCH_DTYPE]
ATTN_IMPLEMENTATION = os.getenv("ATTN_IMPLEMENTATION")
DEVICE_MAP = os.getenv("DEVICE_MAP")
TENSOR_TYPE=os.getenv("TENSOR_TYPE")
MAX_NEW_TOKENS=int(os.getenv("MAX_NEW_TOKENS"))

class Env:
    MODEL_NAME = MODEL_NAME
    PROJECT_TITLE = PROJECT_TITLE
    MODEL_PATH = MODEL_PATH
    TORCH_DTYPE = TORCH_DTYPE
    ATTN_IMPLEMENTATION = ATTN_IMPLEMENTATION
    DEVICE_MAP = DEVICE_MAP
    TENSOR_TYPE = TENSOR_TYPE
    MAX_NEW_TOKENS = MAX_NEW_TOKENS

env = Env()
