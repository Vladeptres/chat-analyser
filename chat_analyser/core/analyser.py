from mistralai import Mistral
import json
from os.path import join as pjoin
import markdown
from chat_analyser.api.models import ConversationAnalysisResponse
from chat_analyser import config as cf

client = Mistral(api_key=cf.API_KEY)


def load_context(context_type: str) -> str:
    if context_type not in cf.AVAILABLE_CONTEXTS:
        raise ValueError(
            f"Context type {context_type} not among existing context. Please choose one among : {cf.AVAILABLE_CONTEXTS}"
        )
    with open(pjoin(cf.CONTEXTS_DIR, context_type + ".md"), "r") as f:
        return markdown.markdown(f.read())


def format_query(
    context_type: str, messages: list[dict[str, str]], users: list[str]
) -> str:
    messages = "\n".join([json.dumps(message) for message in messages])
    users = "\n".join(users)
    return load_context(context_type) + "\n" + messages + "\n" + users


def analyse_chat(
    context_type: str,
    messages: list[dict[str, str]],
    users: list[str],
    model: str = cf.MISTRAL_MODEL,
) -> ConversationAnalysisResponse:
    chat_response = (
        client.chat.complete(
            model=model,
            messages=[
                {"role": "user", "content": format_query(context_type, messages, users)}
            ],
            response_format={
                "type": "json_object",
                "json_schema": ConversationAnalysisResponse.model_json_schema(),
            },
        )
        .choices[0]
        .message.content
    )
    chat_response = chat_response[
        chat_response.index("{") : chat_response.rindex("}") + 1
    ]

    return ConversationAnalysisResponse.model_validate_json(chat_response)
