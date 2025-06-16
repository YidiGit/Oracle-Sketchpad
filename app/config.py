from functools import lru_cache
from pathlib import Path
from pydantic_settings import BaseSettings
from pydantic import Field    
 
class Settings(BaseSettings):
    """All runtime configuration pulled from the environment."""
    # folders
    dataset_dir: Path = Path("Data/dataset")
    assets_dir: Path = Path("Data/assets")
    embedding_path: Path = Path("model_make/animal_embeddings.npy")
    label_path: Path = Path("model_make/animal_labels.npy")

    # performance / tuning
    torch_threads: int = Field(2, env="TORCH_THREADS")

    log_level: str = "INFO"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache
def get_settings() -> Settings:
    """Cache settings so they are evaluated once per process."""
    return Settings()
