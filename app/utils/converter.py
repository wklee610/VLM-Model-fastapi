from app.model.schemas import (
    Conversation,
    Message,
    TextContent,
    ChatCompletionMessage,
)

def openai_msg_convert(
    messages: list[ChatCompletionMessage]
) -> Conversation:
    converted = []

    for msg in messages:
        converted.append(
            Message(
                role=msg.role,
                content=[TextContent(type="text", text=msg.content)]
            )
        )

    return Conversation(root=converted)