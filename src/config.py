import yaml
from pathlib import Path
from pydantic import BaseModel, field_validator

ROOT_DIR = Path(__file__).resolve().parent.parent

vlm_model = "Qwen/Qwen2-VL-7B-Instruct-AWQ"

class DirConfig(BaseModel):
    # Define relative paths
    processed_data: Path =  Path("data/processed")
    raw_data: Path = Path("data/raw")
    embeddings: Path = Path("embeddings")
    indexes: Path = Path("indexes")
    models: Path = Path("models")
    logs: Path = Path("logs")
    notebooks: Path = Path("notebooks")
    scripts: Path = Path("scripts")
    services: Path = Path("services")
    generator: Path = Path("src/generator")
    pipelines: Path = Path("src/pipelines")
    retriever: Path = Path("src/retriever")

    # Create relative paths to the root directory *before* config created
    @field_validator("*", mode="before")            
    @classmethod 
    def resolve_path(cls, val = str | Path):                 
        path = Path(val)
        if not path.is_absolute():
            res_path = ROOT_DIR / path
            res_path.mkdir(parents=True, exist_ok=True)
            return res_path
        return path
    
class ModelConfig(BaseModel):
    vlm_model: str = "Qwen/Qwen3-VL-4B-Thinking-FP8"
    colpali_model: str = "vidore/colqwen2-v1.0"

class RetrievalConfig(BaseModel):
    top_k: int = 3
    
# Master config
class AppConfig(BaseModel):
    dirs: DirConfig = DirConfig()
    models: ModelConfig = ModelConfig()
    retrieval: RetrievalConfig = RetrievalConfig()

def load_config(path = "configs/config.yaml"):
    config_path = ROOT_DIR / path
    if not config_path.exists():
        return AppConfig()
        
    with open(config_path, "r") as f:
        data = yaml.safe_load(f) or {}

    return AppConfig(**data)
