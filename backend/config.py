"""Central configuration, loaded from environment / .env file.

Uses pydantic-settings so every value is typed and validated at startup.
"""
from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Later files win, so .env.local overrides .env for local secrets/overrides.
    model_config = SettingsConfigDict(
        env_file=(".env", ".env.local"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- AWS / Bedrock ---
    aws_region: str = "us-east-1"
    # DeepSeek V3.2 and GLM 4.7 Flash are served directly on Bedrock (no
    # inference profile needed). Both support the Converse API + structured
    # output. GLM-Flash is fast/cheap for high-volume extraction; DeepSeek
    # V3.2 is a stronger reasoner for final synthesis.
    extraction_model_id: str = "zai.glm-4.7-flash"
    synthesis_model_id: str = "deepseek.v3.2"
    embedding_model_id: str = "amazon.titan-embed-text-v2:0"

    # --- Provider selection ---
    # Any name registered in backend.llm.providers: bedrock (default),
    # openai, anthropic, gemini, ollama, moonshot.
    llm_provider: str = "bedrock"

    # Generic model overrides applied to whichever provider is active.
    # Empty means "use that provider's documented default". Bedrock also
    # honours the older EXTRACTION_MODEL_ID / SYNTHESIS_MODEL_ID names.
    extraction_model: str = ""
    synthesis_model: str = ""

    # --- Per-provider credentials (standard env var names) ---
    openai_api_key: str = ""
    openai_base_url: str = ""          # blank = api.openai.com
    anthropic_api_key: str = ""
    google_api_key: str = ""
    ollama_base_url: str = "http://localhost:11434"
    moonshot_api_key: str = ""
    moonshot_base_url: str = "https://api.moonshot.ai/v1"

    # --- Embedding provider (clustering only) ---
    # "bedrock" = Titan; "openai" = any OpenAI-compatible /embeddings endpoint;
    # "local"  = keyless hashed TF-IDF, no network, weaker paraphrase matching.
    embedding_provider: str = "bedrock"
    openai_embedding_api_key: str = ""
    openai_embedding_base_url: str = "https://api.openai.com/v1"
    openai_embedding_model: str = "text-embedding-3-small"

    # --- Ingestion / selection strategy ---
    # Search this many candidates, then select `target_videos` to process.
    # The top `force_deepgram_top_n` (by search rank) are ALWAYS processed:
    # captions if available, else Deepgram. Remaining slots are filled by the
    # next highest-ranked videos that already have captions (free path).
    num_candidates: int = 50
    target_videos: int = 10
    force_deepgram_top_n: int = 5
    default_max_videos: int = 10  # back-compat; maps to target_videos
    transcript_languages: str = "en"  # comma-separated priority list

    # --- Deepgram (transcription fallback for caption-less top videos) ---
    deepgram_api_key: str = ""
    deepgram_model: str = "nova-3"

    # --- Clustering / dedupe ---
    dedupe_similarity_threshold: float = 0.82
    # Lexical TF-IDF cosine runs much lower than semantic cosine for
    # paraphrases, so the "local" provider needs its own default.
    # 0.35 was calibrated on paraphrase pairs: it merged 4/5 true
    # duplicates with 0 false merges across 185 pairs (noise floor 0.25).
    local_dedupe_similarity_threshold: float = 0.35

    # --- Whisper fallback ---
    enable_whisper_fallback: bool = False
    whisper_model_size: str = "large-v3"
    whisper_device: str = "cpu"
    whisper_compute_type: str = "int8"

    # --- Storage ---
    db_path: str = "plan4me.db"

    @property
    def transcript_language_list(self) -> list[str]:
        return [c.strip() for c in self.transcript_languages.split(",") if c.strip()]

    @property
    def deepgram_enabled(self) -> bool:
        return bool(self.deepgram_api_key.strip())


@lru_cache
def get_settings() -> Settings:
    """Cached settings singleton."""
    return Settings()
