from pydantic import BaseModel
from chat_analyser import config as cf

class ConversationAnalysisRequest(BaseModel):
    context_type: str
    model: str = cf.MISTRAL_MODEL
    messages: list[dict[str, str]]
    users: list[str]
    max_attempts: int = 3


class UserFeedback(BaseModel):
    summary: str
    emoji: str


class ConversationAnalysisResponse(BaseModel):
    summary: str
    users_feedback: dict[str, UserFeedback]


class PostContextRequest(BaseModel):
    context_type: str
    context: str


class PostContextResponse(BaseModel):
    available_contexts: list[str]
