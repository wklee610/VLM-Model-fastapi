import uuid
import time
from fastapi import APIRouter, Request
from app.model.schemas import ChatCompletionRequest
from app.service.chat import ChatGenerator
from app.utils.converter import openai_msg_convert

router = APIRouter()


@router.post("")
async def chat_completions(
    body: ChatCompletionRequest,
    request: Request
):
    model_manager = request.app.state.model_manager
    model_container = model_manager.get_model()

    conversation = openai_msg_convert(body.messages)
    generator = ChatGenerator(model_container)

    output = generator.generate(conversation)

    return {
        "id": f"chatcmpl-{uuid.uuid4()}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": body.model,
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": output
            },
            "finish_reason": "stop"
        }]
    }
