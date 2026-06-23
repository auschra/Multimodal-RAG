import yaml
from pathlib import Path
from typing import Any, Dict, Tuple, Type
from pydantic import BaseModel, field_validator
from pydantic_settings import (BaseSettings, SettingsConfigDict, PydanticBaseSettingsSource)

ROOT_DIR = Path(__file__).resolve().parent.parent                   # Relative paths to root

class DirConfig(BaseModel):

    # Pydantic dirs
    processed_data: Path = Path("data/processed")
    raw_data: Path = Path("data/raw")
    embeddings: Path = Path("embeddings")
    logs: Path = Path("logs")
    chunks: Path = Path("data/chunks")
    entities: Path = Path("data/entities")
    graph: Path = Path("data/graph")
    scripts: Path = Path("scripts")
    generator: Path = Path("src/generator")
    pipelines: Path = Path("src/pipelines")
    retriever: Path = Path("src/retriever")
    ingest: Path = Path("src/ingest")

    # Create relative paths to the root directory before config created, if not already
    @field_validator("*", mode="before")
    @classmethod
    def resolve_path(cls, val: str):
 
        path = Path(val)
        if not path.is_absolute():
            res = ROOT_DIR / path
            res.mkdir(parents=True, exist_ok=True)
            return res
        return path

class MineruConfig(BaseModel):
    api_url: str = "http://localhost:8000"
    host_input_dir: Path = Path("/storage/bulk/raw_docs")
    container_input_dir: Path = Path("/input")
    host_output_dir: Path = Path("/datasets/scratch/mineru-output")
    container_output_dir: Path = Path("/output")

class NetworkConfig(BaseModel):
    vllm_api_base: str = "http://localhost:8001/v1"
    qdrant_url: str = "http://localhost:6333"


class ModelConfig(BaseModel):
    vlm_model: str = "Qwen/Qwen3-32B-AWQ"
    colpali_model: str = "vidore/colqwen2-v1.0"
    embedding_model: str = "BAAI/bge-m3"

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="APP_", 
        env_nested_delimiter="__",
        env_file=".env", 
        extra="ignore"
    )

    # config subclass
    dirs: DirConfig = DirConfig()
    mineru: MineruConfig = MineruConfig()
    network: NetworkConfig = NetworkConfig()
    models: ModelConfig = ModelConfig()

    hf_token: str = ""
    top_k: int = 3
    token_limit: int = 1024

    # handling for .yaml variables env > yaml > 
    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: Type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ):
        
        def yaml_settings_source(settings: BaseSettings) -> Dict[str, Any]:
            config_path = ROOT_DIR / "configs" / "config.yaml"
            if config_path.exists():
                with open(config_path) as f:
                    return yaml.safe_load(f) or {}
            return {}
        
        return (init_settings, env_settings, dotenv_settings, yaml_settings_source, file_secret_settings)

config = Settings()