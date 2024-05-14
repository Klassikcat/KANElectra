from typing import Optional
from kan import KANLayer
from torch import (
    nn, 
    Tensor, 
    LongTensor,
    optim
)
import lightning.pytorch as pl
from .models import (
    KANElectraGeneator,
    KANElectraDiscriminator,
)


class ElectraPretrainingHandler(pl.LightningModule):
    def __init__(
        self,
        generator: KANElectraGeneator,
        discriminator: KANElectraDiscriminator,
        optimizer: optim.Optimizer, 
        scheduler: Optional[optim.lr_scheduler.LRScheduler] = None
        ) -> None:
        super().__init__()
        self.optimizer = optimizer
    
    def forward(
        self, 
        input_ids: LongTensor, 
        attention_mask: LongTensor,
        token_type_ids: LongTensor,
    ):
        pass
    
    def configure_optimizers(self) -> nn.Module:
        pass
        
    def training_step(self, batch: Tensor, batch_idx: int) -> Tensor:
        pass
    
    def validateion_step(self, batch: Tensor, batch_idx: int) -> Tensor:
        pass
    
    def test_step(self, batch: Tensor, batch_idx: int) -> Tensor:
        pass
    
    
class ElectraFineTuningHandler(pl.LightningModule):
    def __init__(
        self, 
        model: KANElectraGeneator|KANElectraDiscriminator, 
        prediction_head: nn.Module, 
        optimizer: optim.Optimizer, 
        transforms: Optional[nn.Module] = None
        ) -> None:
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.prediction_head = prediction_head
        self.transforms = transforms
    
    def forward(
        self, 
        input_ids: LongTensor, 
        attention_mask: LongTensor,
        token_type_ids: LongTensor,
    ):
        pass
    
    def configure_optimizers(self) -> nn.Module:
        pass
        
    def training_step(self, batch: Tensor, batch_idx: int) -> Tensor:
        pass
    
    def validateion_step(self, batch: Tensor, batch_idx: int) -> Tensor:
        pass
    
    def test_step(self, batch: Tensor, batch_idx: int) -> Tensor:
        pass