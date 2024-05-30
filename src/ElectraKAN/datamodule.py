from typing import List, Tuple, Set, Optional

import torch
import numpy as np
from transformers import PreTrainedTokenizer
from torch.utils.data import Dataset, DataLoader
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
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self) -> int:
        return len(self.texts)
    
    def dynamic_masking(self, tokens: torch.Tensor) -> Tuple[torch.LongTensor, torch.LongTensor]:
        tokens_to_be_masked = tokens.clone()
        num_tokens = len(tokens)
        masked_indices = torch.bernoulli(torch.full((num_tokens,), 0.15)).bool()
        tokens_to_be_masked[masked_indices] = self.tokenizer.mask_token_id
        return tokens_to_be_masked
    
    def __getitem__(self, idx: int) -> Tuple[torch.LongTensor, torch.LongTensor, torch.LongTensor, torch.LongTensor, torch.LongTensor]:
        input_ids, attention_mask, token_type_ids = tuple(
            self.tokenizer(
                self.texts[idx], 
                return_dict=True, 
                return_attention_mask=True,
                return_token_type_ids=True,
                return_tensors='pt',
                max_length=self.max_length
            )\
            .values()
        )
        masked_input_ids, y_true_ids = self.dynamic_masking(input_ids)
        return masked_input_ids, attention_mask, token_type_ids, input_ids
    
    @classmethod
    def from_csv(cls):
        pass


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
        input_ids, attention_mask, token_type_ids = tuple(
            self.tokenizer(
                text, 
                return_dict=True, 
                return_attention_mask=True,
                return_token_type_ids=True,
                return_tensors='pt',
                max_length=self.max_length
            )\
            .values()
        )
        label_id = self.label_dict[label]
        return input_ids, attention_mask, token_type_ids, torch.tensor(label_id, dtype=torch.long)
    
    @classmethod
    def from_csv(cls):
        pass