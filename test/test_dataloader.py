import sys
from pathlib import Path
from flytekit import FlyteFile
from datasets import load_dataset
from transformers import AutoTokenizer

sys.path.append(str(Path(__file__).parent.parent))

from pipeline.modules import (
    PretrainingDataset, 
    split_sentences,
    
)


def test_pretraining_dataset():
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    dataset = PretrainingDataset(
        file=FlyteFile(path="data/pretraining_dataset.jsonl"),
        tokenizer=tokenizer,
        max_length=128,
    )