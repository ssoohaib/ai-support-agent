# app/config.py
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    database_url: str
    pg_uri: str
    jwt_secret: str
    main_model: str = "gpt-6.1-terra"     # agents, vision, judge
    fast_model: str = "gpt-6-luna"        # router, guard, checks, simple answers

settings = Settings()