import os
import sys
from pathlib import Path
from ruamel.yaml import YAML
from omegaconf import DictConfig
PARENT_PATH = str(Path(os.path.abspath(__file__)).parent.parent)
sys.path.append(PARENT_PATH)
from src.ElectraKAN import ElectraModel

yaml = YAML(typ="safe")
with open(f"{PARENT_PATH}/configs/train.yaml", "r") as f:
    config = DictConfig(yaml.load(f))

model = ElectraModel(config.nn)
