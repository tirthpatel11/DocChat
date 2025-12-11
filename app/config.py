import os
from dotenv import load_dotenv


def load_config() -> None:
    """
    Load environment variables from a .env file if present.

    This should be called once on application startup.
    """
    load_dotenv()


def get_groq_api() -> str:
    """
    Return the Groq API key from the environment.

    The key is expected to be set in an environment variable named
    'GROQ_API_KEY'. This keeps secrets out of source control.
    """
    return os.getenv("GROQ_API_KEY", "")


groq_api_key = get_groq_api()

if not groq_api_key:
    # Avoid printing secrets; only warn when the key is missing.
    print("Warning: GROQ_API_KEY is not set. DocChat will not be able to call Groq LLM APIs.")
