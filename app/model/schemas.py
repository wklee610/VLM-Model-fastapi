from pydantic import (
    BaseModel, 
    Field, 
    RootModel
)
from typing import (
    List, 
    Literal, 
    Union
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