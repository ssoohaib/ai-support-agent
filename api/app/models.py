# app/models.py: the only place models are created
from langchain_openai import ChatOpenAI

from app.config import settings

def openai_model(name: str, effort: str) -> ChatOpenAI:
    return ChatOpenAI(
        model=name,
        use_responses_api=True,          # reliable tool calling with OpenAI's reasoning models
        reasoning={"effort": effort},    # less reasoning = faster and cheaper
    )

fast_model = openai_model(settings.fast_model, "none")
smart_model = openai_model(settings.main_model, "low")