import hydra
from transformers import AutoTokenizer
from omegaconf import DictConfig
try:
    from ElectraKAN import (
        datamodule,
        handlers
    )
except ModuleNotFoundError:
    import sys
    sys.path.append('../')
    from src.ElectraKAN import (
        datamodule,
        handlers
    )


@hydra.main(config_path="../configs", config_name="train")
def main(cfg: DictConfig) -> None:
    pass

