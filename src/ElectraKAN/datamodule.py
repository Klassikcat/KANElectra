import csv
from pathlib import Path
from os import PathLike
from typing import List, Tuple, Set, Optional, Dict, Any

import torch
from pyarrow import parquet as pq
import numpy as np
from transformers import PreTrainedTokenizer, AutoTokenizer
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import lightning.pytorch as ptl


class ElectraKANDataModule(ptl.LightningDataModule):
    def __init__(
        self,
        train_dataset: Dataset,
        val_dataset: Dataset,
        test_dataset: Dataset,
        batch_size: int,
        num_workers: int,
        pin_memory: bool
    ) -> None:
        super().__init__()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False
        )


class StreamingElectraPretrainingDataset(Dataset):
    def __init__(
        self,
        path: PathLike|str,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512,
        chunk_size: int = 1000,
        text_column: str = "text"
    ) -> None:
        super().__init__()
        self.path = path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.chunk_size = chunk_size
        self.text_column = text_column
        self.current_chunk = []
        self.current_chunk_idx = 0
        self.total_samples = self._count_samples()
        
    def _count_samples(self) -> int:
        if str(self.path).endswith('.parquet'):
            return len(pq.read_table(self.path))
        else:
            with open(self.path, 'r') as f:
                return sum(1 for _ in f) - 1  # 헤더 제외
                
    def _load_next_chunk(self):
        if str(self.path).endswith('.parquet'):
            table = pq.read_table(self.path, skip=self.current_chunk_idx, take=self.chunk_size)
            self.current_chunk = table.column(self.text_column).to_pylist()
        else:
            with open(self.path, 'r') as f:
                reader = csv.reader(f)
                next(reader)  # 헤더 스킵
                for _ in range(self.current_chunk_idx):
                    next(reader)
                self.current_chunk = [next(reader)[0] for _ in range(min(self.chunk_size, self.total_samples - self.current_chunk_idx))]
        self.current_chunk_idx += len(self.current_chunk)

    def __len__(self) -> int:
        return self.total_samples

    def dynamic_masking(self, tokens: torch.Tensor) -> torch.Tensor:
        tokens_to_be_masked: torch.Tensor = tokens.clone()
        end_of_token: int = int(torch.where(tokens == self.tokenizer.sep_token_id)[0]) - 1
        masked_indices = torch.bernoulli(torch.full((end_of_token,), 0.15)).bool()
        if masked_indices[0] == True:
            masked_indices[0] = False
        padded_masked_indices: torch.Tensor = F.pad(masked_indices, mode='constant', value=False, pad=(self.max_length - end_of_token, 0))
        tokens_to_be_masked = torch.where(padded_masked_indices == True, self.tokenizer.mask_token_id, tokens_to_be_masked)
        return tokens_to_be_masked

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.LongTensor]:
        if idx >= self.current_chunk_idx or idx < self.current_chunk_idx - len(self.current_chunk):
            self.current_chunk_idx = (idx // self.chunk_size) * self.chunk_size
            self._load_next_chunk()
        
        local_idx = idx - (self.current_chunk_idx - len(self.current_chunk))
        text = self.current_chunk[local_idx]
        
        tokenized = tuple(
            self.tokenizer(
                text,
                return_attention_mask=True,
                return_token_type_ids=True,
                return_tensors='pt',
                max_length=self.max_length,
                padding="max_length",
                truncation=True
            ).values()
        )
        input_ids, token_type_ids, attention_mask = tokenized
        masked_input_ids = self.dynamic_masking(input_ids.squeeze(0))
        return masked_input_ids, attention_mask.squeeze(0), token_type_ids.squeeze(0), input_ids.squeeze(0)

    @classmethod
    def from_csv(
        cls,
        path: PathLike|str,
        tokenizer: PreTrainedTokenizer|str,
        text_row: int,
        text_b_row: Optional[int] = None,
        max_length: int = 512,
        chunk_size: int = 1000
    ):
        if isinstance(tokenizer, str):
            tokenizer_instance = AutoTokenizer.from_pretrained(tokenizer)
        else:
            tokenizer_instance = tokenizer
        return cls(path, tokenizer_instance, max_length, chunk_size)
    
    @classmethod
    def from_parquet(
        cls,
        path: PathLike|str,
        tokenizer: PreTrainedTokenizer|str,
        text_column: str,
        max_length: int = 512,
        chunk_size: int = 1000
    ):
        if isinstance(tokenizer, str):
            tokenizer_instance = AutoTokenizer.from_pretrained(tokenizer)
        else:
            tokenizer_instance = tokenizer
        return cls(path, tokenizer_instance, max_length, chunk_size, text_column)


class StreamingElectraClassificationDataset(Dataset):
    def __init__(
        self,
        path: PathLike|str,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512,
        chunk_size: int = 1000,
        text_column: str = "text",
        label_column: str = "label"
    ) -> None:
        super().__init__()
        self.path = path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.chunk_size = chunk_size
        self.text_column = text_column
        self.label_column = label_column
        self.current_chunk = []
        self.current_chunk_idx = 0
        self.total_samples = self._count_samples()
        self.label_dict = self._build_label_dict()
        
    def _count_samples(self) -> int:
        if str(self.path).endswith('.parquet'):
            return len(pq.read_table(self.path))
        else:
            with open(self.path, 'r') as f:
                return sum(1 for _ in f) - 1  # 헤더 제외

    def _build_label_dict(self) -> Dict[str, int]:
        if str(self.path).endswith('.parquet'):
            table = pq.read_table(self.path)
            labels = set(table.column(self.label_column).to_pylist())
        else:
            with open(self.path, 'r') as f:
                reader = csv.reader(f)
                next(reader)  # 헤더 스킵
                labels = set(row[1] for row in reader)
        return {label: idx for idx, label in enumerate(sorted(labels))}
                
    def _load_next_chunk(self):
        if str(self.path).endswith('.parquet'):
            table = pq.read_table(self.path, skip=self.current_chunk_idx, take=self.chunk_size)
            texts = table.column(self.text_column).to_pylist()
            labels = table.column(self.label_column).to_pylist()
            self.current_chunk = list(zip(texts, labels))
        else:
            with open(self.path, 'r') as f:
                reader = csv.reader(f)
                next(reader)  # 헤더 스킵
                for _ in range(self.current_chunk_idx):
                    next(reader)
                self.current_chunk = [next(reader) for _ in range(min(self.chunk_size, self.total_samples - self.current_chunk_idx))]
        self.current_chunk_idx += len(self.current_chunk)

    def __len__(self) -> int:
        return self.total_samples

    def __getitem__(self, idx: int) -> Tuple[torch.LongTensor, torch.LongTensor, torch.LongTensor, torch.LongTensor]:
        if idx >= self.current_chunk_idx or idx < self.current_chunk_idx - len(self.current_chunk):
            self.current_chunk_idx = (idx // self.chunk_size) * self.chunk_size
            self._load_next_chunk()
        
        local_idx = idx - (self.current_chunk_idx - len(self.current_chunk))
        text, label = self.current_chunk[local_idx]
        
        tokenized = tuple(
            self.tokenizer(
                text,
                return_attention_mask=True,
                return_token_type_ids=True,
                return_tensors='pt',
                max_length=self.max_length,
                padding="max_length",
                truncation=True
            ).values()
        )
        input_ids, token_type_ids, attention_mask = tokenized
        label_id = self.label_dict[label]
        return input_ids.squeeze(0), attention_mask.squeeze(0), token_type_ids.squeeze(0), torch.tensor(label_id, dtype=torch.long)

    @classmethod
    def from_csv(
        cls,
        path: PathLike|str,
        tokenizer: PreTrainedTokenizer|str,
        text_row: int,
        label_row: int,
        text_b_row: Optional[int] = None,
        max_length: int = 512,
        chunk_size: int = 1000
    ):
        if isinstance(tokenizer, str):
            tokenizer_instance = AutoTokenizer.from_pretrained(tokenizer)
        else:
            tokenizer_instance = tokenizer
        return cls(path, tokenizer_instance, max_length, chunk_size)

    @classmethod
    def from_parquet(
        cls,
        path: PathLike|str,
        tokenizer: PreTrainedTokenizer|str,
        text_column: str,
        label_column: str,
        max_length: int = 512,
        chunk_size: int = 1000
    ):
        if isinstance(tokenizer, str):
            tokenizer_instance = AutoTokenizer.from_pretrained(tokenizer)
        else:
            tokenizer_instance = tokenizer
        return cls(path, tokenizer_instance, max_length, chunk_size, text_column, label_column)

