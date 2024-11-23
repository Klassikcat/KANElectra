import csv
from os import PathLike
from typing import List, Tuple, Set, Optional

import torch
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


class ElectraPretrainingDataset(Dataset):
    def __init__(
        self,
        texts: List[str],
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512
    ) -> None:
        super().__init__()
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def dynamic_masking(self, tokens: torch.Tensor) -> torch.Tensor:
        tokens_to_be_masked: torch.Tensor = tokens.clone()
        end_of_token: int = int(torch.where(tokens == self.tokenizer.sep_token_id)[0].item()) - 1
        masked_indices = torch.bernoulli(torch.full((end_of_token,), 0.15)).bool()
        if masked_indices[0] == True:
            masked_indices[0] = False
            masked_indices[1] = True
        padded_masked_indices: torch.Tensor = F.pad(masked_indices, mode='constant', value=False, pad=(self.max_length - end_of_token, 0))
        tokens_to_be_masked[padded_masked_indices] = self.tokenizer.mask_token_id
        return tokens_to_be_masked

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.LongTensor, torch.Tensor]:
        tokenized: Tuple[torch.LongTensor, torch.LongTensor, torch.LongTensor] = tuple(
            self.tokenizer(
                self.texts[idx],
                return_attention_mask=True,
                return_token_type_ids=True,
                return_tensors='pt',
                max_length=self.max_length,
                padding="max_length",
                truncation=True
            )\
            .values()
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
        max_length: int = 512
        ):
        with open(path, 'r') as f:
            reader = csv.reader(f)
            dataset = []
            for row in reader:
                text = row[text_row]
                label = row[text_row]
                text_b = row[text_b_row] if text_b_row else None
                dataset.append(text, label, text_b) if text_b else dataset.append(text, label)
        return cls(dataset, tokenizer, max_length)


class ElectraClassificationDataset(Dataset):
    def __init__(
        self,
        texts_and_labels: List[str],
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512,
        labels: Optional[List[str]|Set[str]] = None,
    ) -> None:
        super().__init__()
        self.texts_and_labels = texts_and_labels
        self.labels = labels if labels else set([i[1] for i in texts_and_labels])
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label_dict = {
            v: k for k, v in enumerate(self.labels)
        }

    def __len__(self) -> int:
        return len(self.texts_and_labels)

    def __getitem__(self, idx: int) -> Tuple[torch.LongTensor, torch.LongTensor, torch.LongTensor, torch.LongTensor]:
        text, label = self.texts_and_labels[idx]
        input_ids, token_type_ids, attention_mask = tuple(
            self.tokenizer(
                text,
                return_attention_mask=True,
                return_token_type_ids=True,
                return_tensors='pt',
                max_length=self.max_length,
                padding="max_length",
                truncation=True
            )\
            .values()
        )
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
        max_length: int = 512
    ):
        tokenizer = tokenizer if isinstance(tokenizer, PreTrainedTokenizer) else AutoTokenizer.from_pretrained(tokenizer)
        dataset = []
        labels = []
        with open(path, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                text = row[text_row]
                text_b_row = row[text_b_row] if text_b_row else None
                label = row[label_row]
                dataset.append((text, label, text_b_row)) if text_b_row else dataset.append((text, label))
                labels.append(label)
        return cls(dataset, tokenizer, max_length=max_length, labels=list(set(labels)))
