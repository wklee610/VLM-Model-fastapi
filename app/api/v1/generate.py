from fastapi import (
    APIRouter, 
    Request
)
from app.model.schemas import Conversation
from app.service.chat import ChatGenerator

router = APIRouter()

@router.post("")
async def generate(conversation: Conversation, request: Request):
    try:
        model_manager = request.app.state.model_manager
        llava_model_container = model_manager.get_llava_model()
        chat_generator = ChatGenerator(llava_model_container)
        result = chat_generator.generate(conversation) 
        return {"Answer": result}
    
    except Exception as e:
        return {"Internal Server Error"}
