import yaml
from pydantic import BaseModel


class AudioFormatConfig(BaseModel):
    channels: int
    sample_rate: int


class AudioConfig(BaseModel):
    input: AudioFormatConfig
    output: AudioFormatConfig


class Config(BaseModel):
    audio: AudioConfig


def load_config(path: str) -> Config:
    with open(path, "r", encoding="utf-8") as file:
        data = yaml.safe_load(file)
    return Config(**data)


CONFIG = load_config(path="config.yaml")
