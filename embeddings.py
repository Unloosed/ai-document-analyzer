import logging
from typing import List, Union

from chromadb.api.types import EmbeddingFunction, Embeddings
from openai import APIConnectionError, APIError, AuthenticationError, OpenAI

from config import EMBEDDING_MODEL, OPENROUTER_API_KEY, OPENROUTER_BASE_URL

logger = logging.getLogger(__name__)

# Dedicated client for embeddings
embedding_client = OpenAI(
    api_key=OPENROUTER_API_KEY,
    base_url=OPENROUTER_BASE_URL,
)

class OpenAIEmbeddingFunction(EmbeddingFunction):
    def __init__(self, model_name: str = EMBEDDING_MODEL):
        self.model_name = model_name

    def __call__(self, input: Union[str, List[str]]) -> Embeddings:
        if not input:
             return []

        # Ensure input is a list if it's a single string
        texts = [input] if isinstance(input, str) else input

        try:
            response = embedding_client.embeddings.create(
                model=self.model_name,
                input=texts,
            )
            return [item.embedding for item in response.data]
        except AuthenticationError as e:
            logger.error("Authentication error with OpenAI API: %s", e)
            raise
        except APIConnectionError as e:
            logger.error("Network issue connecting to OpenAI API: %s", e)
            raise
        except APIError as e:
            logger.error("OpenAI API returned an error: %s", e)
            raise
        except Exception as e:
            logger.error("Unexpected error during embedding generation: %s", e)
            raise
