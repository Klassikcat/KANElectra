from pathlib import Path
from typing import Optional, Tuple, Any, Dict, List
import torch
from torch import (
    Tensor,
    LongTensor,
    FloatTensor
)
import torch.nn.functional as F
import lightning.pytorch as pl
from torchmetrics.classification import Accuracy, Precision, Recall, FBetaScore
from omegaconf import DictConfig
from .modules import (
    ElectraGenerator,
    ElectraDiscriminator
)


class ElectraModel(pl.LightningModule):
    def __init__(self, config: DictConfig):
        super(ElectraModel, self).__init__()
        self.generator = ElectraGenerator(**config.generator)
        self.discriminator = ElectraDiscriminator(**config.discriminator)
        self.config = config

        self.accuracy = Accuracy(task='multiclass', num_labels=2)
        self.precision = Precision(task='multiclass', num_labels=2)
        self.recall = Recall(task='multiclass', num_labels=2)
        self.f1 = FBetaScore(task='multiclass', num_labels=2, beta=1.0)

    def forward(self, input_ids: LongTensor, attention_mask: LongTensor, token_type_ids: LongTensor) -> Tuple[Tensor, Tensor]:
        generator_logits = self.generator(input_ids, attention_mask, token_type_ids)
        # Sample new tokens based on generator logits
        sampled_ids = torch.argmax(generator_logits, dim=-1)  # Greedy sampling

        # Replace masked tokens in input_ids with sampled_ids
        input_ids_with_generated = input_ids.clone()
        mask_token_indices = input_ids == self.config.mask_token_id
        input_ids_with_generated[mask_token_indices] = sampled_ids[mask_token_indices]

        # Discriminator processes the replaced input_ids
        discriminator_logits = self.discriminator(input_ids_with_generated, attention_mask, token_type_ids)
        return generator_logits, discriminator_logits

    def training_step(self, batch: Tuple[LongTensor, LongTensor, LongTensor], batch_idx: int, optimizer_idx: int) -> Tensor:
        input_ids, attention_mask, token_type_ids = batch
        generator_logits, discriminator_logits = self(input_ids, attention_mask, token_type_ids)

        if optimizer_idx == 0:
            generator_loss = self.generator_loss(generator_logits, input_ids)
            self.log('train_generator_loss', generator_loss)
            return generator_loss

        elif optimizer_idx == 1:
            discriminator_loss = self.discriminator_loss(discriminator_logits, input_ids)
            self.log('train_discriminator_loss', discriminator_loss)
            return discriminator_loss
        else:
            raise ValueError(f'Invalid optimizer index: {optimizer_idx}')

    def validation_step(self, batch: Tuple[LongTensor, LongTensor, LongTensor], batch_idx: int) -> None:
        input_ids, attention_mask, token_type_ids = batch
        generator_logits, discriminator_logits = self(input_ids, attention_mask, token_type_ids)

        generator_loss = self.generator_loss(generator_logits, input_ids)
        discriminator_loss = self.discriminator_loss(discriminator_logits, input_ids)

        self.log('val_generator_loss', generator_loss)
        self.log('val_discriminator_loss', discriminator_loss)

        preds = torch.argmax(discriminator_logits, dim=1)
        self.log('val_accuracy', self.accuracy(preds, input_ids))
        self.log('val_precision', self.precision(preds, input_ids))
        self.log('val_recall', self.recall(preds, input_ids))
        self.log('val_f1', self.f1(preds, input_ids))

    def configure_optimizers(self) -> Tuple[List[torch.optim.Optimizer], List[Any]]:
        generator_optimizer = torch.optim.Adam(self.generator.parameters(), lr=self.config.generator_lr)
        discriminator_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=self.config.discriminator_lr)
        return [generator_optimizer, discriminator_optimizer], []

    def generator_loss(self, generator_logits: LongTensor, input_ids: LongTensor) -> Tensor:
        # Only calculate loss for masked tokens
        mask_token_indices = input_ids == self.config.mask_token_id
        logits_for_masked = generator_logits[mask_token_indices]
        labels_for_masked = input_ids[mask_token_indices]
        return F.cross_entropy(logits_for_masked, labels_for_masked)

    def discriminator_loss(self, discriminator_logits: LongTensor, input_ids: LongTensor) -> Tensor:
        generated_labels = self.create_discriminator_labels(input_ids, discriminator_logits)
        return F.cross_entropy(discriminator_logits.view(-1, 2), generated_labels.view(-1))

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        checkpoint['generator_state_dict'] = self.generator.state_dict()
        checkpoint['discriminator_state_dict'] = self.discriminator.state_dict()

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])

    def create_discriminator_labels(self, input_ids: LongTensor, generated_logits: LongTensor) -> torch.Tensor:
        # Compare original and generated logits
        labels = (input_ids != generated_logits).long()
        return labels
