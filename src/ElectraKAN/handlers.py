import torch
from collections.abc import Mapping
from typing import cast

from torch import (
    Tensor,
    LongTensor,
)
from typing_extensions import override
import torch.nn.functional as F
import lightning.pytorch as pl
from torchmetrics.classification import Accuracy, Precision, Recall, FBetaScore
from omegaconf import DictConfig
from .modules import ElectraGenerator, ElectraDiscriminator


class ElectraModel(pl.LightningModule):
    def __init__(self, config: DictConfig):
        super(ElectraModel, self).__init__()
        self.generator = ElectraGenerator(**config.generator)
        self.discriminator = ElectraDiscriminator(**config.discriminator)
        self.config = config
        self._automatic_optimization = False

        self.accuracy = Accuracy(task="multiclass", num_classes=2)
        self.precision = Precision(task="multiclass", num_classes=2)
        self.recall = Recall(task="multiclass", num_classes=2)
        self.f1 = FBetaScore(task="multiclass", num_classes=2, beta=1.0)

    def locate_mask(self, input_ids: Tensor) -> Tensor:
        return input_ids == self.config.mask_token_id

    @override
    def forward(
        self,
        input_ids: LongTensor,
        attention_mask: LongTensor,
        token_type_ids: LongTensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        # Output: Tensor[batch_size, seq_length, vocab_size]
        generator_logits: Tensor = self.generator(
            input_ids, attention_mask, token_type_ids
        )
        # Sample new tokens based on generator logits
        # generator_logits shape: [batch_size, seq_length, vocab_size]
        batch_size, seq_length, vocab_size = generator_logits.shape
        probs = generator_logits.exp()  # Convert log probs to probs
        probs_2d = probs.view(-1, vocab_size)  # Reshape to 2D for multinomial
        sampled_ids = torch.multinomial(probs_2d, num_samples=1).view(batch_size, seq_length)

        # Replace masked tokens in input_ids with sampled_ids
        input_ids_with_generated = input_ids.clone()
        mask_token_indices = input_ids == self.config.mask_token_id
        input_ids_with_generated[mask_token_indices] = sampled_ids[mask_token_indices]

        # Discriminator processes the replaced input_ids
        discriminator_logits = self.discriminator(
            input_ids_with_generated, attention_mask, token_type_ids
        )
        return generator_logits, discriminator_logits, sampled_ids

    @override
    def training_step(
        self,
        batch: tuple[LongTensor, LongTensor, LongTensor, LongTensor],
        batch_idx: int,
    ) -> None:
        masked_input_ids, attention_mask, token_type_ids, input_ids = batch
        generator_logits, discriminator_logits, sampled_ids = self(
            masked_input_ids, attention_mask, token_type_ids
        )

        optimizers = cast(list[torch.optim.Optimizer], self.trainer.optimizers)
        generator_optimizer, discriminator_optimizer = optimizers

        mask_token_indices = self.locate_mask(masked_input_ids)
        generator_loss = self.generator_loss(
            generator_logits, input_ids, mask_token_indices
        )
        self.log(
            "train_generator_loss",
            generator_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )
        generator_loss.backward()
        generator_optimizer.step()
        generator_optimizer.zero_grad()

        generated_labels = self.create_discriminator_labels(
            input_ids, sampled_ids, mask_token_indices
        )

        discriminator_loss = self.discriminator_loss(
            discriminator_logits, generated_labels
        )
        self.log(
            "train_discriminator_loss",
            discriminator_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )

        discriminator_loss.backward()
        discriminator_optimizer.step()  # noqa
        discriminator_optimizer.zero_grad()  # noqa

        preds = torch.argmax(discriminator_logits, dim=-1)
        self.log(
            "train_accuracy", self.accuracy(preds, generated_labels), on_epoch=True
        )
        self.log(
            "train_precision", self.precision(preds, generated_labels), on_epoch=True
        )
        self.log("train_recall", self.recall(preds, generated_labels), on_epoch=True)
        self.log(
            "train_f1",
            self.f1(preds, generated_labels),
            on_epoch=True,
            prog_bar=True,
            on_step=True,
        )

    @torch.no_grad()
    def validation_step(
        self,
        batch: tuple[LongTensor, LongTensor, LongTensor, LongTensor],
        batch_idx: int,
    ) -> None:
        masked_input_ids, attention_mask, token_type_ids, input_ids = batch
        generator_logits, discriminator_logits, sampled_ids = self(
            masked_input_ids, attention_mask, token_type_ids
        )

        mask_token_indices = self.locate_mask(masked_input_ids)
        generator_loss = self.generator_loss(
            generator_logits, input_ids, mask_token_indices
        )

        generated_labels = self.create_discriminator_labels(
            input_ids, sampled_ids, mask_token_indices
        )

        discriminator_loss = self.discriminator_loss(
            discriminator_logits, generated_labels
        )

        self.log(
            "val_generator_loss",
            generator_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "val_discriminator_loss",
            discriminator_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )

        preds = torch.argmax(discriminator_logits, dim=-1)
        self.log("val_accuracy", self.accuracy(preds, generated_labels), on_epoch=True)
        self.log(
            "val_precision", self.precision(preds, generated_labels), on_epoch=True
        )
        self.log("val_recall", self.recall(preds, generated_labels), on_epoch=True)
        self.log(
            "val_f1", self.f1(preds, generated_labels), on_epoch=True, prog_bar=True
        )

    def configure_optimizers(self):
        generator_optimizer = torch.optim.Adam(
            self.generator.parameters(), lr=self.config.generator_lr
        )
        discriminator_optimizer = torch.optim.Adam(
            self.discriminator.parameters(), lr=self.config.discriminator_lr
        )
        return [generator_optimizer, discriminator_optimizer], []

    def generator_loss(
        self,
        generator_logits: LongTensor,
        input_ids: LongTensor,
        mask_token_indices: Tensor,
    ) -> Tensor:
        # Only calculate loss for masked tokens
        logits_for_masked = generator_logits[mask_token_indices]
        labels_for_masked = input_ids[mask_token_indices]
        return F.cross_entropy(logits_for_masked, labels_for_masked)

    def discriminator_loss(
        self, discriminator_logits: LongTensor, generated_labels: Tensor | LongTensor
    ) -> Tensor:
        return F.cross_entropy(
            discriminator_logits.view(-1, 2), generated_labels.view(-1)
        )

    def on_save_checkpoint(self, checkpoint: dict[str, object]) -> None:
        checkpoint["generator_state_dict"] = self.generator.state_dict()
        checkpoint["discriminator_state_dict"] = self.discriminator.state_dict()

    def on_load_checkpoint(self, checkpoint: dict[str, object]) -> None:
        generator_state = cast(
            Mapping[str, torch.Tensor], checkpoint["generator_state_dict"]
        )
        discriminator_state = cast(
            Mapping[str, torch.Tensor], checkpoint["discriminator_state_dict"]
        )
        self.generator.load_state_dict(generator_state)
        self.discriminator.load_state_dict(discriminator_state)

    def create_discriminator_labels(
        self,
        input_ids: LongTensor,
        sampled_ids: Tensor | LongTensor,
        mask_token_indices: Tensor,
    ) -> torch.Tensor:
        labels = torch.zeros_like(input_ids)
        labels[mask_token_indices] = (
            input_ids[mask_token_indices] != sampled_ids[mask_token_indices]
        ).long()
        return labels
