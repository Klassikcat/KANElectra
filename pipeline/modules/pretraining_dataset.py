from flytekit import FlyteFile
from typing import List
from torch.utils.data import Dataset


class PretrainingDataset(Dataset):
    def __init__(self, files: FlyteFile, tokenizer: PreTrainedTokenizer, max_length: int):
        self.files = files
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __getitem__(self, idx: int):
        return super().__getitem__(idx)
