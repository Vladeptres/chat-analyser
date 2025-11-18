from fastapi import FastAPI
from chat_analyser.api.models import (
    PostContextRequest,
    PostContextResponse,
    ConversationAnalysisRequest,
    ConversationAnalysisResponse,
    UserFeedback,
)
from chat_analyser import core, config as cf

app = FastAPI()


@app.post("/chat/")
def analyse_chat(request: ConversationAnalysisRequest) -> ConversationAnalysisResponse:
    attempts = 0
    last_error = None
    while attempts < request.max_attempts:
        try:
            return core.analyse_chat(
                request.context_type, request.messages, request.users
            )
        except ValueError as e:
            last_error = e
            attempts += 1
    return ConversationAnalysisResponse(
        summary=str(last_error) if last_error else "Max attempts exceeded",
        users_feedback={
            user: UserFeedback(summary=user, emoji="") for user in request.users
        },
    )


@app.post("/context/")
def post_context(request: PostContextRequest) -> PostContextResponse:
    core.write_context(request.context_type, request.context)
    return PostContextResponse(available_contexts=cf.AVAILABLE_CONTEXTS)
