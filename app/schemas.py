from typing import Optional

from pydantic import BaseModel


class ParseRequest(BaseModel):
    text: str


class ParseResponse(BaseModel):
    intent: str
    date: Optional[str] = None
    time: Optional[str] = None
    place: Optional[str] = None