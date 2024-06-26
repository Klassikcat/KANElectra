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
    ElectraDiscriminator,
    Classifier
)
from .kan import KAN


class AccelerationHandler(pl.LightningModule):
    def __init__(self):
        super().__init__()
        
    def info(
        self,
        log_dict: Dict[str, Any],
        stage: str
    ):
        for k, v in log_dict.items():
            self.log(f"{stage}_{k}", v)
    
    def configure_optimizers(self) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]:
        optimizer = getattr(torch.optim, self.config.optimizer.name)(
            self.parameters(), **self.config.optimizer.params
        )
        scheduler = getattr(torch.optim.lr_scheduler, self.config.scheduler.name)(
            optimizer, **self.config.scheduler.params
        )
        return [optimizer], [scheduler]
    
    def pass_tensors_and_update_dynamic_axes(
            self,
            dummy_inputs: torch.Tensor | Dict[str, torch.Tensor] | Tuple[torch.Tensor]
    ) -> Tuple[List[str], Dict[str, Dict[int, str]]]:
        if isinstance(dummy_inputs, torch.Tensor):
            _ = self(dummy_inputs)
            dynamic_axes = {'input': {0: 'batch_size'}}
        elif isinstance(dummy_inputs, dict):
            dummy_values = tuple(dummy_inputs.values())
            _ = self(*dummy_values)
            dynamic_axes = {k: {0: 'batch_size'} for k in list(dummy_inputs.keys())}
        elif isinstance(dummy_inputs, tuple):
            print(dummy_inputs)
            _ = self(*dummy_inputs)
            dynamic_axes = {'input_{}'.format(i): {0: 'batch_size'} for i in range(len(dummy_inputs))}
        else:
            raise TypeError(
                "dummy_inputs must be a torch.Tensor, tuple of torch.Tensor, or dict {string: torch.Tensor},"
                f"but got {type(dummy_inputs)}.")
        input_keys = list(dynamic_axes.keys())
        dynamic_axes.update({'output': {0: 'batch_size'}})
        return input_keys, dynamic_axes

    def to_onnx(
            self,
            dirpath: str|Path,
            dummy_inputs: Tuple[torch.Tensor, ...] | Dict[str, torch.Tensor] | torch.Tensor,
            **kwargs: Any
    ) -> None:
        input_keys, dynamic_axes = self.pass_tensors_and_update_dynamic_axes(dummy_inputs)
        torch.onnx.export(
            self,
            tuple(dummy_inputs.values()),
            dirpath,
            export_params=True,
            opset_version=17,
            do_constant_folding=True,
            input_names=input_keys,
            output_names=['output'],
            dynamic_axes=dynamic_axes
        )


class ElectraPretrainingHandler(AccelerationHandler):
    def __init__(self, config: DictConfig) -> None:
        super().__init__()
        self.generator = ElectraGenerator(**config.generator) 
        self.discriminator = ElectraDiscriminator(**config.discriminator) 
    
    def forward(
        self, 
        input_ids: LongTensor, 
        attention_mask: LongTensor,
        token_type_ids: LongTensor,
    ):
        out_gen = self.generator(input_ids, attention_mask, token_type_ids)
        out_gen_tokenzed = torch.argmax(out_gen, dim=-1).long()
        out_disc = self.discriminator(out_gen_tokenzed, attention_mask, token_type_ids)
        return out_disc
        
    def training_step(self, batch: Tensor, batch_idx: int) -> Tensor:
        input_ids, attention_masks, token_type_ids, label_ids = batch # unpack the batch. label_id will be generated by the dataloader automatically.
        out = self(input_ids, attention_masks, token_type_ids)
        loss = F.cross_entropy(out, label_ids)
        # TODO: add more metrics
        self.info({"train_loss": loss})
        return loss
    
    def validateion_step(self, batch: Tensor, batch_idx: int) -> Tensor:
        input_ids, attention_masks, token_type_ids, label_ids = batch # unpack the batch. label_id will be generated by the dataloader automatically.
        out = self(input_ids, attention_masks, token_type_ids)
        loss = F.cross_entropy(out, label_ids)
        # TODO: add more metrics
        self.info({"train_loss": loss})
        return loss
    
    def test_step(self, batch: Tensor, batch_idx: int) -> Tensor:
        input_ids, attention_masks, token_type_ids, label_ids = batch # unpack the batch. label_id will be generated by the dataloader automatically.
        out = self(input_ids, attention_masks, token_type_ids)
        loss = F.cross_entropy(out, label_ids)
        
        self.info({"train_loss": loss})
        return loss
    
    
class ElectraMaskedLM(AccelerationHandler):
    def __init__(
        self, config: DictConfig
    ):
        super().__init__()
        self.generator = ElectraGenerator(**config.nn)
        
    def forward(
        self,
        input_ids: LongTensor,
        attention_mask: Optional[LongTensor] = None,
        token_type_ids: Optional[LongTensor] = None,
        return_tokens: bool = False
    ) -> FloatTensor|LongTensor:
        out_gen = self.generator(input_ids, attention_mask, token_type_ids)
        if return_tokens:
            output = torch.argmax(out_gen, dim=-1).long()
        else:
            output = out_gen
        return output


class ElectraEmbeddingGenerator(AccelerationHandler):
    def __init__(
        self, config: DictConfig
    ):
        super().__init__()
        # TODO: Implement Sentence-Transformer.


class ClassificationHandler(AccelerationHandler):
    def __init__(
        self, config: DictConfig, num_labels: int, beta: float = 1.0
    ):
        super().__init__()
        self.beta = beta
        self.discriminator = ElectraDiscriminator(**config.nn)
        self.accuracy = Accuracy(
            num_classes=num_labels,
            ignore_index=0,
            task="binary" if num_labels <= 2 else 'multiclass',
            )
        self.precision = Precision(
            num_classes=num_labels,
            ignore_index=0,
            task="binary" if num_labels <= 2 else 'multiclass',
        )
        self.recall = Recall(
            num_classes=num_labels,
            ignore_index=0,
            task="binary" if num_labels <= 2 else 'multiclass',
        )
        self.f_beta= FBetaScore(
            num_classes=num_labels,
            ignore_index=0,
            task="binary" if num_labels <= 2 else 'multiclass',
            beta=beta
        )

    def calculate_score(
        self,
        y_hat: Tensor,
        y_true: Tensor
    ):
        accuracy_score = self.accuracy(y_hat, y_true)
        precision_score = self.accuracy(y_hat, y_true)
        recall_score = self.accuracy(y_hat, y_true)
        fbeta_score = self.accuracy(y_hat, y_true)
        return {
            "accuracy": accuracy_score,
            "precision": precision_score,
            "recall": recall_score,
            f"f_{self.beta}": fbeta_score
        }
    
    def training_step(self, batch: Tensor, batch_idx: int) -> Tensor:
        input_ids, attention_masks, token_type_ids, label_ids = batch
        out = self(input_ids, attention_masks, token_type_ids)
        loss = F.cross_entropy(out, label_ids)
        score_dict = self.calculate_score(out, label_ids)
        score_dict["loss"] = loss
        self.info(score_dict, "train")
        return loss
    
    def validation_step(self, batch: Tensor, batch_idx: int) -> Tensor:
        input_ids, attention_masks, token_type_ids, label_ids = batch
        out = self(input_ids, attention_masks, token_type_ids)
        loss = F.cross_entropy(out, label_ids)
        score_dict = self.calculate_score(out, label_ids)
        score_dict["loss"] = loss
        self.info(score_dict, "val")
        return loss
    
    def test_step(self, batch: Tensor, batch_idx: int) -> Tensor:
        input_ids, attention_masks, token_type_ids, label_ids = batch
        out = self(input_ids, attention_masks, token_type_ids)
        loss = F.cross_entropy(out, label_ids)
        score_dict = self.calculate_score(out, label_ids)
        score_dict["loss"] = loss
        self.info(score_dict, "test")
        return loss


class ElectraSequenceClassifier(ClassificationHandler):
    def __init__(
        self,
        config: DictConfig,
        num_labels: int,
        beta: float = 1.0
    ):
        super().__init__(config, num_labels, beta)
        self.classifier = Classifier(
            hidden_dim=config.nn.hidden_dim,
            num_labels=num_labels
            )
    
    def forward(
        self, 
        input_ids: LongTensor, 
        attention_mask: LongTensor,
        token_type_ids: LongTensor,
    ) -> Tensor:
        seq_out = self.discriminator(input_ids, attention_mask, token_type_ids)
        x = seq_out[:, 0, :]  # take <s> token
        output = self.classifier(x)
        return output
    


class ElectraTokenClassifier(ClassificationHandler):
    def __init__(
        self, 
        config: DictConfig,
        num_labels: int,
        beta: float = 1.0
    ):
        super().__init__(config, num_labels, beta)
        self.classifier = KAN(width=[config.nn.hidden_dim, num_labels])
        
    def forward(
        self, 
        input_ids: LongTensor, 
        attention_mask: LongTensor,
        token_type_ids: LongTensor,
    ) -> Tensor:
        seq_out = self.discriminator(input_ids, attention_mask, token_type_ids)
        dropouted_seq_output = F.dropout(seq_out, p=self.config.nn.dropout_p)
        output = self.classifier(dropouted_seq_output)
        return output
    