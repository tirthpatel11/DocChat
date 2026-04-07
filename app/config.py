import os
from dotenv import load_dotenv


def load_config() -> None:
    load_dotenv()


def get_groq_api() -> str:
    return os.getenv("GROQ_API_KEY", "")
