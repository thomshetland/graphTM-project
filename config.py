from typing import List, Literal, Optional

from pydantic import BaseModel
from ruamel.yaml import YAML

class ModelConfig(BaseModel):
    epochs: int
    number_of_clauses: int
    T: int
    s: float
    depth: int
    hypervector_bits: int

class NodeConfig(BaseModel):
    symbols: List[str]

class EdgeConfig(BaseModel):
    symbols: List[str]

class VectorConfig(BaseModel):
    hv_size: int
    hv_bits: int
    msg_size: int
    msg_bits: int

class GameConfig(BaseModel):
    board_size: int

class AppConfig(BaseModel):
    model: ModelConfig
    node: NodeConfig
    edge: EdgeConfig
    vector: VectorConfig
    game: GameConfig
    

def load_config(path: str = "config.yaml") -> AppConfig:
    yaml = YAML(typ="safe")
    with open(path, "r") as f:
        data = yaml.load(f)
    return AppConfig(**data)

config = load_config()