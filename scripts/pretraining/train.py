import hydra
from transformers import AutoTokenizer
from omegaconf import DictConfig
from typing import List
import lightning.pytorch as pl
import datasets
try:
    from ElectraKAN.datamodule import ElectraKANDataModule, ElectraPretrainingDataset
    from ElectraKAN.handlers import ElectraModel
    from ElectraKAN import callbacks
except ModuleNotFoundError:
    import os
    import sys
    from pathlib import Path
    PARENT_PATH = str(Path(os.path.abspath(__file__)).parent.parent.parent)
    sys.path.append(PARENT_PATH)
    from src.ElectraKAN.datamodule import ElectraKANDataModule, ElectraPretrainingDataset
    from src.ElectraKAN.handlers import ElectraModel
    from src.ElectraKAN import callbacks


@hydra.main(config_path="../../configs", config_name="train")
def main(cfg: DictConfig) -> None:
    datamodule = get_dataloader(
        tokenizer_path=cfg.tokenizer_name,
        dataset_config=cfg.datasets,
        datamodule_config=cfg.datamodule
    )
    model = ElectraModel(cfg.nn)
    callback_lst = get_callbacks(cfg.trainer.callbacks)
    del cfg.trainer.callbacks
    trainer = pl.Trainer(**cfg.trainer, callbacks=callback_lst)
    trainer.fit(model, datamodule=datamodule)
    trainer.test(model, datamodule=datamodule)

def get_callbacks(
    callbacks_config: DictConfig
    ) -> List[pl.Callback]:
    callback_lst: List[pl.Callback] = []
    for config in callbacks_config:
        try:
            callback = getattr(pl.callbacks, config.name)(**config.params)
        except ModuleNotFoundError:
            callback = getattr(callbacks, config.name)(**config.params)
        callback_lst.append(callback)
    return callback_lst


def get_dataloader(
    tokenizer_path: str,
    dataset_config: DictConfig,
    datamodule_config: DictConfig
    ) -> ElectraKANDataModule:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    train_dataset = ElectraPretrainingDataset.from_csv(
        path=dataset_config.train.path,
        tokenizer=tokenizer,
        max_length=dataset_config.train.max_length,
        text_row=dataset_config.train.text_row
    )
    val_dataset = ElectraPretrainingDataset.from_csv(
        path=dataset_config.val.path,
        tokenizer=tokenizer,
        max_length=dataset_config.val.max_length,
        text_row=dataset_config.val.text_row
    )
    test_dataset = ElectraPretrainingDataset.from_csv(
        path=dataset_config.test.path,
        tokenizer=tokenizer,
        max_length=dataset_config.test.max_length,
        text_row=dataset_config.test.text_row
    )
    datamodule = ElectraKANDataModule(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        batch_size=datamodule_config.batch_size,
        num_workers=datamodule_config.num_workers,
        pin_memory=datamodule_config.pin_memory
    )
    return datamodule


def download_datasets(
    dataset_config: DictConfig
) -> None:
    raw_dataset = datasets.load_dataset(
        path=dataset_config.raw_data.path,
        split=dataset_config.raw_data.split
    )
    raw_dataset.save_to_disk(dataset_config.raw_data.path)


if __name__ == '__main__':
    main()
