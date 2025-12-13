from mistralai import Mistral
from os.path import join as pjoin
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
        return f.read()


def format_user_prompt(
    users: list[str],
    conversations: dict[int, dict[str, str]],
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
    messages: list[dict[str, str]],
    model: str = cf.MISTRAL_MODEL,
    chunk_size: int = 30,
) -> ConversationAnalysisResponse:
    """Call a Mistral LLM to analyse the messages of users given a context type.
    The model outputs relevant informations defined in the system prompt (which depends on the context type), in a structured format.

    Messages are processed in chunks each to handle large conversations efficiently.

    Here is the expected format for the messages list:
    ```
    [
        {"user": "Alice", "content": "Hi Bob, please find attached the documents for the meeting."},
        {"user": "Bob", "content": "Thank you Alice. I'm looking forward to see your presentation."},
    ]
    ```

    Args:
        context_type (str): The name of the context to load.
        users (list[str]): The list of users registered in the chat.
        messages (list[dict[str, str]]): The messages list.
        model (str, optional): The name of the Mistral model to call. Defaults to cf.MISTRAL_MODEL.

    Returns:
        ConversationAnalysisResponse: A pydantic ConversationAnalysisResponse object containing the analysis results.
    """
    chunks = [messages[i : i + chunk_size] for i in range(0, len(messages), chunk_size)]

    # If only one chunk, use original logic
    if len(chunks) == 1:
        system_prompt = load_system_prompt(context_type)
        user_prompt = format_user_prompt(users, messages)

        api_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        with Mistral(api_key=cf.API_KEY) as client:
            chat_response = (
                client.chat.complete(
                    model=model,
                    messages=api_messages,
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

    # Process multiple chunks
    chunk_analyses = []

    with Mistral(api_key=cf.API_KEY) as client:
        for i, chunk in enumerate(chunks):
            # Modify system prompt to indicate chunk analysis
            base_system_prompt = load_system_prompt(context_type)
            chunk_system_prompt = f"{base_system_prompt}\n\nIMPORTANT: You are analyzing chunk {i + 1} of {len(chunks)} from a larger conversation. This chunk contains messages {chunk_size * i}-{chunk_size * (i + 1)} out of the full conversation (messages 0-{len(messages)}). Focus on analyzing only this subset of messages."

            user_prompt = format_user_prompt(users, chunk)

            api_messages = [
                {"role": "system", "content": chunk_system_prompt},
                {"role": "user", "content": user_prompt},
            ]

            # Get analysis for this chunk
            chat_response = (
                client.chat.complete(
                    model=model,
                    messages=api_messages,
                    stream=False,
                    response_format={
                        "type": "json_object",
                        "json_schema": ConversationAnalysisResponse.model_json_schema(),
                    },
                )
                .choices[0]
                .message.content
            )

            # Clean and parse response
            chat_response = chat_response[
                chat_response.index("{") : chat_response.rindex("}") + 1
            ]

            chunk_analysis = ConversationAnalysisResponse.model_validate_json(
                chat_response
            )
            chunk_analyses.append(chunk_analysis)

        # Create final merge request
        merge_system_prompt = f"{load_system_prompt(context_type)}\n\nIMPORTANT: You are now merging {len(chunk_analyses)} separate analyses of chunks from the same conversation into a single comprehensive analysis. Combine the insights, patterns, and findings from all chunks to provide a unified analysis of the entire conversation."

        # Format chunk analyses for the merge prompt
        chunk_summaries = []
        for i, analysis in enumerate(chunk_analyses):
            chunk_summaries.append(
                f"## Chunk {i + 1} Analysis\n{analysis.model_dump_json(indent=2)}"
            )

        merge_user_prompt = (
            f"## Users list\n{users}\n\n## Chunk Analyses to Merge\n"
            + "\n\n".join(chunk_summaries)
        )

        merge_messages = [
            {"role": "system", "content": merge_system_prompt},
            {"role": "user", "content": merge_user_prompt},
        ]

        # Get final merged analysis
        final_response = (
            client.chat.complete(
                model=model,
                messages=merge_messages,
                stream=False,
                response_format={
                    "type": "json_object",
                    "json_schema": ConversationAnalysisResponse.model_json_schema(),
                },
            )
            .choices[0]
            .message.content
        )

        # Clean and parse final response
        final_response = final_response[
            final_response.index("{") : final_response.rindex("}") + 1
        ]

        return ConversationAnalysisResponse.model_validate_json(final_response)
