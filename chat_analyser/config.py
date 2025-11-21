import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = str(Path(__file__).parent.parent)

API_KEY = os.getenv("API_KEY")
MISTRAL_MODEL = os.getenv("MISTRAL_MODEL", "mistral-tiny-latest")

CONTEXTS_DIR = os.path.join(BASE_DIR, "chat_analyser", "core", "contexts")
AVAILABLE_CONTEXTS = [f.split(".")[0] for f in os.listdir(CONTEXTS_DIR)]
