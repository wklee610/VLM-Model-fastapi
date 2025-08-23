from fastapi import APIRouter
from app.api.v1.healthz import router as healthz
from app.api.v1.generate import router as generate

router = APIRouter()

router.include_router(healthz, prefix='/healthz', tags=["healthz"])
router.include_router(generate, prefix='/generate', tags=["generate"])