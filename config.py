from utils.file_ops import get_env_variable

OPENROUTER_API_KEY = get_env_variable(
    "OPENROUTER_API_KEY", required=False, default="", redact=True
)
OPENROUTER_BASE_URL = get_env_variable(
    "OPENROUTER_BASE_URL",
    required=False,
    default="https://openrouter.ai/api/v1",
    redact=False,
)
EMBEDDING_MODEL = get_env_variable(
    "EMBEDDING_MODEL",
    required=False,
    default="openai/text-embedding-3-small",
    redact=False,
)
CHAT_MODEL = get_env_variable(
    "CHAT_MODEL", required=False, default="openrouter/free", redact=False
)
