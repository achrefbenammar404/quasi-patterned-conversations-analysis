import os
from functools import lru_cache
from pydantic_settings import BaseSettings
from typing import Literal , List
import os 
class Settings(BaseSettings):
    """
    Settings class for this application.
    Utilizes the BaseSettings from pydantic for environment variables.
    """
    GOOGLE_API_KEY : str 
    ENVIRONMENT: Literal["local", "dev", "prod"]
    MISTRAL_API_KEY : str 
    class Config:
        env_file = ".env"
        extra = "ignore"  # Ignore any extra environment variables

@lru_cache(maxsize=None)
def get_settings():
    """Function to get and cache settings.
    The settings are cached to avoid repeated disk I/O."""
    environment = os.getenv("ENVIRONMENT", "local")
    if environment == "local":
        return Settings(_env_file=".env")  # type: ignore
    elif environment == "dev":
        return Settings(_env_file=".env.dev")  # type: ignore
    elif environment == "prod":
        return Settings(_env_file=".env.prod")  # type: ignore
    else:
        raise ValueError(f"Invalid environment: {environment}")