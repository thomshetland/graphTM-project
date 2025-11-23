from typing import List, Literal, Optional

from pydantic import BaseModel
from ruamel.yaml import YAML


class GameConfig(BaseModel):
    board_size: int

class AppConfig(BaseModel):
    game: GameConfig
    

def load_config(path: str = "config.yaml") -> AppConfig:
    yaml = YAML(typ="safe")
    with open(path, "r") as f:
        data = yaml.load(f)
    return AppConfig(**data)

config = load_config()