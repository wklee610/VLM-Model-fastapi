import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.dont_write_bytecode = True      # no __pycache__

import gc
import multiprocessing as mp

from fastapi import FastAPI
import uvicorn
from contextlib import asynccontextmanager

from app.api import router as api
from app.common.env import env
from app.manager.gpu_manager import GPUManager
from app.manager.model_manager import ModelManager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # GPU check
    gpu_manager = GPUManager()
    device = gpu_manager.is_gpu_available()
    tensor_parallel_size = gpu_manager.get_tensor_parallel_size()
    
    # Model
    model_manager = ModelManager(
        env, 
        device, 
        tensor_parallel_size
    )
    model_manager.load()
    app.state.model_manager = model_manager
    yield
    
    # EXIT
    gpu_manager.free_gpu_resources()
    del app.state.model_manager
    del model_manager
    del gpu_manager
    gc.collect()
    mp.active_children()    # warning 제거

app = FastAPI(
    lifespan=lifespan,
    title=env.PROJECT_TITLE
)

app.include_router(api, prefix="/api")

if __name__ == "__main__":
    uvicorn.run(
        "run:app", 
        host="0.0.0.0", 
        port=8080,
        # reload=True
    )