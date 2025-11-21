from mistralai import Mistral
from os.path import join as pjoin
import markdown
from chat_analyser.api.models import ConversationAnalysisResponse
from chat_analyser import config as cf


def load_system_prompt(context_type: str) -> str:
    """Loads the context to be used as a system prompt.

    Args:
        context_type (str): The name of the context to load.

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


def format_user_prompt(
    users: list[str], conversations: dict[int, dict[str, str]]
) -> str:
    """Format the user prompt given the list of users, and the conversations dictionary.

    Args:
        users (list[str]): The list of users.
        conversations (dict[int, dict[str, str]]): The conversations dictionary

    Returns:
        str: The formatted user prompt.
    """

    template = (
        """## Users list\n{users}\n\n"""
        """## Conversations\n{json_messages}"""
    )

    formatted_prompt = template.format(users=users, json_messages=conversations)
    return formatted_prompt


def analyse_chat(
    context_type: str,
    users: list[str],
    conversations: list[dict[str, str]],
    model: str = cf.MISTRAL_MODEL,
) -> ConversationAnalysisResponse:
    """Call a Mistral LLM to analyse the conversations of users given a context type.
    The model outputs relevant informations defined in the system prompt (which depends on the context type), in a structured format.

    Here is the expected format for the conversations dict:
    ```
    {
        0: {"user": "Alice", "content": "Hi Bob, please find attached the documents for the meeting."},
        1: {"user": "Bob", "content": "Thank you Alice. I'm looking forward to see your presentation."},
    }
    ```

    Args:
        context_type (str): The name of the context to load.
        users (list[str]): The list of users registered in the chat.
        conversations (list[dict[str, str]]): The conversations dictionary.
        model (str, optional): The name of the Mistral model to call. Defaults to cf.MISTRAL_MODEL.

    Returns:
        ConversationAnalysisResponse: A pydantic ConversationAnalysisResponse object containing the analysis results.
    """
    system_prompt: str = load_system_prompt(context_type)
    user_prompt: str = format_user_prompt(users, conversations)

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
