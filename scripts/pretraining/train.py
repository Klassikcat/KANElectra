import hydra
import csv
from transformers import AutoTokenizer
from omegaconf import DictConfig
from typing import List
import lightning.pytorch as pl
from pyarrow import parquet as pq

from ElectraKAN.datamodule import ElectraKANDataModule, StreamingElectraPretrainingDataset
from ElectraKAN.handlers import ElectraModel
from ElectraKAN import callbacks

from data_processing import DatasetDownloader, DatasetPreprocessor


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
    
    # 파일 확장자에 따라 적절한 데이터셋 생성 메서드 선택
    if str(dataset_config.train.path).endswith('.parquet'):
        train_dataset = StreamingElectraPretrainingDataset.from_parquet(
            path=dataset_config.train.path,
            tokenizer=tokenizer,
            max_length=dataset_config.train.max_length,
            text_column="text",
            chunk_size=datamodule_config.chunk_size
        )
        val_dataset = StreamingElectraPretrainingDataset.from_parquet(
            path=dataset_config.val.path,
            tokenizer=tokenizer,
            max_length=dataset_config.val.max_length,
            text_column="text",
            chunk_size=datamodule_config.chunk_size
        )
        test_dataset = StreamingElectraPretrainingDataset.from_parquet(
            path=dataset_config.test.path,
            tokenizer=tokenizer,
            max_length=dataset_config.test.max_length,
            text_column="text",
            chunk_size=datamodule_config.chunk_size
        )
    else:
        train_dataset = StreamingElectraPretrainingDataset.from_csv(
            path=dataset_config.train.path,
            tokenizer=tokenizer,
            max_length=dataset_config.train.max_length,
            text_row=dataset_config.train.text_row,
            chunk_size=datamodule_config.chunk_size
        )
        val_dataset = StreamingElectraPretrainingDataset.from_csv(
            path=dataset_config.val.path,
            tokenizer=tokenizer,
            max_length=dataset_config.val.max_length,
            text_row=dataset_config.val.text_row,
            chunk_size=datamodule_config.chunk_size
        )
        test_dataset = StreamingElectraPretrainingDataset.from_csv(
            path=dataset_config.test.path,
            tokenizer=tokenizer,
            max_length=dataset_config.test.max_length,
            text_row=dataset_config.test.text_row,
            chunk_size=datamodule_config.chunk_size
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


def preprocess_datasets(
    dataset_config: DictConfig
) -> None:
    downloader = DatasetDownloader(dataset_config)
    preprocessor = DatasetPreprocessor(tokenizer_path=dataset_config.tokenizer_name)
    datasets = downloader()
    for dataset_name, dataset in datasets:
        array = preprocessor(dataset)
        with open(f"data/{dataset_name}.jsonl", "w") as f:
            for item in array:
                f.write(orjson.dumps(item).decode("utf-8") + "\n")


if __name__ == '__main__':
    main()
