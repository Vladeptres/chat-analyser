from chat_analyser import config as cf
from os.path import join as pjoin


def write_context(context_type: str, context: str):
    with open(pjoin(cf.CONTEXTS_DIR, context_type + ".md"), "w") as f:
        f.write(context)
    cf.AVAILABLE_CONTEXTS.append(context_type)
