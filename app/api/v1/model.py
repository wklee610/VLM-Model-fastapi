from fastapi import APIRouter
from typing import Dict

router = APIRouter()

@router.get("")
def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": "VARCO-VISION-2.0-14B",
                "object": "model",
                "owned_by": "Hajun Lee"
            }
        ]
    }