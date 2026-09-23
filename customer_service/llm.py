"""大模型工厂：全项目唯一的 LLM 入口，换模型只改这里或 .env"""

import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()


def get_llm(**kwargs) -> ChatOpenAI:
    return ChatOpenAI(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url=os.getenv(
            "DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
        ),
        model=os.getenv("MODEL_NAME", "qwen-plus"),
        temperature=0.3,
        **kwargs,
    )
