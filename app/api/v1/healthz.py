from fastapi import APIRouter
from typing import Dict

router = APIRouter()

@router.get("")
def health_check() -> Dict[str, str]:
    return {"status": "ok"}