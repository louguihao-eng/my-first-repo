# =============================================================================
# config.py �?全局配置中心
# 所有配置项统一在这里定义，通过 .env 文件或环境变量注�?# 使用 Pydantic Settings 实现类型安全 + 自动校验
# =============================================================================

import os
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # --- LLM ---
    chat_model_name: str = "deepseek-v4-pro"
    embedding_model_name: str = "text-embedding-v4"
    deepseek_api_key: str = ""
    dashscope_api_key: str = ""

    # --- 数据库 ---
    lamp_db_path: str = "data/lamp.db"

    # --- ChromaDB ---
    chroma_persist_dir: str = "chroma_db"
    chroma_collection_name: str = "lamp_knowledge"
    chunk_size: int = 300
    chunk_overlap: int = 30
    rag_top_k: int = 5

    # --- 知识库 ---
    knowledge_dir: str = "data/knowledge"


    # --- RAG---
    rag_rewrite_enabled: bool = True
    rag_rerank_enabled: bool = True
    rag_hybrid_search_enabled: bool = True

    # --- LangFuse 可观测性（可选，未配置时自动跳过）---
    langfuse_enabled: bool = False
    langfuse_public_key: str = ''
    langfuse_secret_key: str = ''
    langfuse_host: str = 'https://cloud.langfuse.com'
    model_config = {
        "env_file": ".env",
        "env_prefix": "LAMP_",
        "extra": "ignore",
    }


settings = Settings()
