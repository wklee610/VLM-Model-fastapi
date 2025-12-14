from fastapi import APIRouter
from app.api.v1.healthz import router as healthz
from app.api.v1.generate import router as generate
from app.api.v1.model import router as models

router = APIRouter()

router.include_router(healthz, prefix='/healthz', tags=["healthz"])
router.include_router(generate, prefix='/chat/completions', tags=["generate"])
router.include_router(models, prefix='/models', tags=["models"])