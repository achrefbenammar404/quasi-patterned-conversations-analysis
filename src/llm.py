from typing import List, Dict, Any
from abc import ABC, abstractmethod
from functools import lru_cache
import google.generativeai as genai
from mistralai import Mistral
from src.config import get_settings
from src.utils.utils import exponential_backoff

settings = get_settings()

class LLM(ABC):
    def __init__(self, model_name: str, model_provider: str):
        """ 
        Args:
            model_name (str): The name of the model.
            model_provider (str): The provider or company behind the model
        """
        self.model_name = model_name
        self.model_provider = model_provider
        self.client = None

    @abstractmethod
    def get_response(self, messages: List[Dict[str , str ]]) -> str:
        """
        Args:
            messages (list): A list of messages (conversation history).

        Returns:
            str: The generated response.
        """
        pass

    def get_info(self) -> Dict[str, Any]:
        """
        Abstract method to return info of the LLM model.
        Returns:
            Dict[str, Any]: info.
        """
        return {
            "model_name": self.model_name,
            "model_provider": self.model_provider,
        }


class MistralLLM(LLM):
    def __init__(self, model_name: str):
        super().__init__(model_name, model_provider="Mistral")
        self.client = Mistral(api_key=settings.MISTRAL_API_KEY)

    @exponential_backoff(retries=10, backoff_in_seconds=2, max_backoff=16)
    def get_response(self,  messages: List[Dict[str , str ]]) -> str:
        """
        Cached method to get a response from the Mistral model.
        """
        # Convert the input list to a hashable tuple for caching compatibility
        message_tuple = tuple((m['role'], m['content']) for m in messages)
        response = self.client.chat.complete(
            model=self.model_name,
            messages=list(messages),  # Convert back to a list for the client
        )
        return response.choices[0].message.content


class GoogleLLM(LLM):
    def __init__(self, model_name: str):
        super().__init__(model_name, "Google")
        genai.configure(api_key=settings.GOOGLE_API_KEY)
        self.client = genai.GenerativeModel(model_name=self.model_name)

    @exponential_backoff(retries=10, backoff_in_seconds=2, max_backoff=16)
    def get_response(self, messages: List[Dict[str , str ]]) -> str:
        """
        Cached method to get a response from the Google model.
        """
        # Modify messages for compatibility with the Google model
        modified_messages = []
        for message in messages:
            role = message['role']
            if role == 'assistant':
                role = 'model'
            elif role == 'system':
                role = 'user'
            modified_message = {
                'role': role,
                'parts': message['content'],  # Changing 'content' to 'parts'
            }
            modified_messages.append(modified_message)

        if len(modified_messages) > 0:
            chat = self.client.start_chat(history=modified_messages[:-1])
            response = chat.send_message(modified_messages[-1], stream=False)
        else:
            chat = self.client.start_chat(history=[])
            response = chat.send_message(modified_messages[0], stream=False)
        return response.text


class CollectionLLM:
    llm_collection: Dict[str, LLM] = {
        "gemini-1.5-pro": GoogleLLM(model_name="gemini-1.5-pro"),
        "gemini-1.5-flash": GoogleLLM(model_name="gemini-1.5-flash"),
        "open-mixtral-8x22b": MistralLLM("open-mixtral-8x22b"),
    }
