from pydantic import (
    BaseModel, 
    Field, 
    RootModel
)
from typing import (
    List, 
    Literal, 
    Union,
    Optional
)

class ImageContent(BaseModel):
    type: Literal["image"]
    url: str


class TextContent(BaseModel):
    type: Literal["text"]
    text: str


class Message(BaseModel):
    role: str
    content: List[Union[ImageContent, TextContent]]


class Conversation(RootModel):
    root: List[Message] = Field(..., description="Message List")


class ChatCompletionMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatCompletionMessage]
    stream: Optional[bool] = False