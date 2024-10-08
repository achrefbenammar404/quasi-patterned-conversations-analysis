from typing import List, Dict, Any
from abc import ABC, abstractmethod
import google.generativeai as genai
from mistralai import Mistral
from src.config import get_settings
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
    async def get_response(
        self, messages: List[str]
    ) -> str:
        """
        Args:
            messages (list):  A list of messages (conversation history).

        Returns:
            str: The generated response.
        """
        pass

    async def get_info(self) -> Dict[str, Any]:
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
        
    def get_response(
        self,
        messages: List[Dict[str, Any]], 
    ) -> str:

        response = self.client.chat.complete(
            model=self.model_name, 
            messages=messages
        )
        return response.choices[0].message.content


class GoogleLLM(LLM):
    def __init__(self, model_name: str):
        super().__init__(model_name, "Google")
        genai.configure(api_key=settings.GOOGLE_API_KEY)
        self.client = genai.GenerativeModel(model_name=self.model_name)
    
    def get_response(
        self,
        messages: List[str], 
    ) -> str:
        modified_messages = []

        for message in messages:
            # Modify the keys as requested
            role = message['role']
            if role == 'assistant':
                role = 'model'
            elif role == 'system' : 
                role = 'user'
            modified_message = {
                'role': role,
                'parts': message['content']  # Changing 'content' to 'parts'
            }
            modified_messages.append(modified_message)
        if len(modified_messages) > 0:
            chat = self.client.start_chat(history=modified_messages[:-1])
            response = chat.send_message(modified_messages[-1], stream=False)
        else:
            chat = self.client.start_chat(history=[])
            response = chat.send_message(modified_messages[0], stream=False)
        return response.text



class CollectionLLM : 
    llm_collection : Dict[str , LLM]= {
            "gemini-1.5-pro": GoogleLLM(model_name="gemini-1.5-pro") ,   
            "gemini-1.5-flash" : GoogleLLM(model_name="gemini-1.5-flash") , 
            "open-mixtral-8x22b" : MistralLLM("open-mixtral-8x22b")
    }




