import os
import json
import torch
from distutils.util import strtobool
from dotenv import load_dotenv

load_dotenv(".env.varco")

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
SYSTEM_PROMPT_FILE = os.getenv("SYSTEM_PROMPT_FILE")
FEW_SHOTS_FILE = os.getenv("FEW_SHOTS_FILE")
USE_VLLM = strtobool(os.getenv("USE_VLLM"))
MAX_MODEL_LEN = int(os.getenv("MAX_MODEL_LEN"))
MAX_NUM_BATCHED_TOKENS = int(os.getenv("MAX_NUM_BATCHED_TOKENS"))
TEMPERATURE = float(os.getenv("TEMPERATURE"))
TOP_P = float(os.getenv("TOP_P"))

def read_txt(file_path: str) -> str:
    if not file_path or not os.path.exists(file_path):
        return ""
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read().strip()

def read_json(file_path: str):
    if not file_path or not os.path.exists(file_path):
        return []
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

SYSTEM_PROMPT = read_txt(SYSTEM_PROMPT_FILE)
FEW_SHOTS = read_json(FEW_SHOTS_FILE)

class Env:
    MODEL_NAME = MODEL_NAME
    PROJECT_TITLE = PROJECT_TITLE
    MODEL_PATH = MODEL_PATH
    TORCH_DTYPE = TORCH_DTYPE
    ATTN_IMPLEMENTATION = ATTN_IMPLEMENTATION
    DEVICE_MAP = DEVICE_MAP
    TENSOR_TYPE = TENSOR_TYPE
    MAX_NEW_TOKENS = MAX_NEW_TOKENS
    SYSTEM_PROMPT_FILE = SYSTEM_PROMPT_FILE
    FEW_SHOTS_FILE = FEW_SHOTS_FILE
    SYSTEM_PROMPT = SYSTEM_PROMPT
    FEW_SHOTS = FEW_SHOTS
    USE_VLLM = USE_VLLM
    MAX_MODEL_LEN = MAX_MODEL_LEN
    MAX_NUM_BATCHED_TOKENS = MAX_NUM_BATCHED_TOKENS
    TEMPERATURE = TEMPERATURE
    TOP_P = TOP_P

env = Env()
