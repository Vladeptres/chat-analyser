from mistralai import Mistral
from os.path import join as pjoin
import markdown
from chat_analyser.api.models import ConversationAnalysisResponse
from chat_analyser import config as cf
from typing import Literal


def load_system_prompt(context_type: Literal["party", "work"]) -> str:
    """Loads the context to be used as a system prompt.

    Args:
        context_type (Literal["party", "work"]): The name of the context to load.

    Raises:
        ValueError: If the requested context type does not exist.

    Returns:
        str: The loaded context as a string.
    """
    if context_type not in cf.AVAILABLE_CONTEXTS:
        raise ValueError(
            f"Context type {context_type} not among existing context. Please choose one among : {cf.AVAILABLE_CONTEXTS}"
        )
    with open(pjoin(cf.CONTEXTS_DIR, context_type + ".md"), "r") as f:
        return markdown.markdown(f.read())


def format_query(conversations: list[dict[str, str]]) -> str:
    prompt = "Here are the messages sent:\n{json_messages}".format(
        json_messages=conversations
    )
    return prompt


def analyse_chat(
    context_type: str,
    conversations: list[dict[str, str]],
    model: str = cf.MISTRAL_MODEL,
) -> ConversationAnalysisResponse:
    system_prompt: str = load_system_prompt(context_type)
    user_prompt: str = format_query(conversations)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    with Mistral(api_key=cf.API_KEY) as client:
        chat_response = (
            client.chat.complete(
                model=model,
                messages=messages,
                stream=False,
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
