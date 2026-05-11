from __future__ import annotations

import logging
from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings loaded from environment variables."""

    database_url: str = Field(
        default="postgresql+asyncpg://user:password@localhost:5432/clinical_copilot",
        validation_alias="DATABASE_URL",
    )
    environment: Literal["development", "staging", "production"] = Field(
        default="development",
        validation_alias="ENVIRONMENT",
    )
    log_level: str = Field(default="INFO", validation_alias="LOG_LEVEL")
    secret_key: str = Field(default="change-me-in-production", validation_alias="SECRET_KEY")
    cors_origins: list[str] = Field(
        default_factory=lambda: ["http://localhost:3000"],
        validation_alias="CORS_ORIGINS",
    )
    openai_api_key: str = Field(default="", validation_alias="OPENAI_API_KEY")
    pinecone_api_key: str = Field(default="", validation_alias="PINECONE_API_KEY")
    pinecone_index_name: str = Field(default="clinical-copilot", validation_alias="PINECONE_INDEX_NAME")
    pinecone_environment: str = Field(default="us-east-1-aws", validation_alias="PINECONE_ENVIRONMENT")
    cohere_api_key: str = Field(default="", validation_alias="COHERE_API_KEY")
    ncbi_email: str = Field(default="", validation_alias="NCBI_EMAIL")
    ncbi_api_key: str | None = Field(default=None, validation_alias="NCBI_API_KEY")
    embedding_model: str = Field(default="text-embedding-3-large", validation_alias="EMBEDDING_MODEL")
    embedding_dimensions: int = Field(default=3072, validation_alias="EMBEDDING_DIMENSIONS")
    reranker_model: str = Field(default="rerank-v3.5", validation_alias="RERANKER_MODEL")
    reranker_top_n: int = Field(default=5, validation_alias="RERANKER_TOP_N")
    retrieval_top_k: int = Field(default=20, validation_alias="RETRIEVAL_TOP_K")
    chunk_size_tokens: int = Field(default=512, validation_alias="CHUNK_SIZE_TOKENS")
    chunk_overlap_tokens: int = Field(default=50, validation_alias="CHUNK_OVERLAP_TOKENS")
    pubmed_max_per_query: int = Field(default=200, validation_alias="PUBMED_MAX_PER_QUERY")

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, value: object) -> list[str] | object:
        """Parse comma-separated CORS origins from env files.

        Args:
            value: Raw value supplied by pydantic-settings.

        Returns:
            Either a parsed list of origins or the original value for Pydantic.

        Raises:
            ValueError: If the environment value is an empty string.
        """
        if isinstance(value, str):
            origins = [origin.strip() for origin in value.split(",") if origin.strip()]
            if not origins:
                raise ValueError("CORS_ORIGINS must include at least one origin")
            return origins
        return value

    @model_validator(mode="after")
    def warn_on_default_secret(self) -> Settings:
        """Warn when non-development environments use the default secret.

        Args:
            None.

        Returns:
            The validated settings instance.

        Raises:
            None.
        """
        if self.environment != "development" and self.secret_key == "change-me-in-production":
            logging.getLogger(__name__).warning(
                "Default SECRET_KEY is configured outside development"
            )
        return self

    def missing_retrieval_settings(self) -> list[str]:
        """Return required retrieval settings that are empty.

        Args:
            None.

        Returns:
            Names of missing settings required for online retrieval.

        Raises:
            None.
        """
        missing = []
        if not self.openai_api_key:
            missing.append("OPENAI_API_KEY")
        if not self.pinecone_api_key:
            missing.append("PINECONE_API_KEY")
        if not self.cohere_api_key:
            missing.append("COHERE_API_KEY")
        return missing


@lru_cache
def get_settings() -> Settings:
    """Return cached application settings.

    Args:
        None.

    Returns:
        Loaded and validated settings.

    Raises:
        pydantic.ValidationError: If environment variables are invalid.
    """
    return Settings()
