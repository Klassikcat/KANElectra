from flytekit import task, Resources, FlyteFile
from typing import List, Tuple, Dict, Any

import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint

from ElectraKAN import ElectraModel, OnnxCompiler
from ElectraKAN.datamodule import ElectraPretrainingDataset



@task(cache=True, cache_version="1.0", limits=Resources(cpu="8", mem="64Gi", gpus="4"))
def train_model(
    train_dataset: List[FlyteFile],
    val_dataset: List[FlyteFile],
    test_dataset: List[FlyteFile],
    batch_size: int,
) -> Dict[str, Any]:
    trainer = pl.Trainer(
        max_epochs=10,
        logger=WandbLogger(project="ElectraKAN"),
        callbacks=[
            ModelCheckpoint(
                monitor="val_loss", 
                mode="min",
                save_last=True,
                save_top_k=1,
                save_on_train_epoch_end=True,
                save_onnx=True,
                save_tensorrt_engine=True,
            ), OnnxCompiler(
                save_onnx=True,
                save_tensorrt_engine=True,
                min_shape=(1, 512),
                opt_shape=(16, 512),
                max_shape=(32, 512),
                save_mixed_precision=True,
            )],
            accelerator="cuda",
            devices="auto",
            precision="16-mixed",
        )
    model = ElectraModel()
    return