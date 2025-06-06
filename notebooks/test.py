import sys
from omegaconf import DictConfig
from pathlib import Path
import tqdm
sys.path.append(str(Path(__file__).parent.parent))
from pyarrow import parquet as pq
from scripts.pretraining.data_processing import DatasetDownloader, DatasetPreprocessor

downloader = DatasetDownloader(dataset_config=DictConfig({
    "raw_data": {
        "dataset_name": "wikipedia",
        "dataset_version": "20220301.en",
        "dataset_split": {
            "train": "train[:1%]",
            "val": "train[1%:2%]",
            "test": "train[:1%]"
        }
    }
}))
preprocessor = DatasetPreprocessor(tokenizer_path="google/electra-base-discriminator")
datasets = downloader()

for dataset_name, dataset in tqdm.tqdm(datasets.items()):
    array = preprocessor(dataset)
    pq.write_table(pq.Table.from_pylist(array), f"data/{dataset_name}.parquet")