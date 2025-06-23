import tqdm
import asyncio
import json
from typing import Dict, Any, List, Generator
from omegaconf import DictConfig
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer


class DatasetDownloader:
    def __init__(self, dataset_config: DictConfig):
        self.dataset_config = dataset_config

    async def download_dataset(self, dataset_name: str, dataset_version: str, dataset_split: str) -> Dataset:
        return load_dataset(dataset_name, dataset_version, split=dataset_split)

    async def download_datasets(self) -> Dict[str, Dataset]:
        tasks = [
            self.download_dataset(
                dataset_name=self.dataset_config.raw_data.dataset_name,
                dataset_version=self.dataset_config.raw_data.dataset_version,
                dataset_split=self.dataset_config.raw_data.dataset_split.train
            ),
            self.download_dataset(
                dataset_name=self.dataset_config.raw_data.dataset_name,
                dataset_version=self.dataset_config.raw_data.dataset_version,
                dataset_split=self.dataset_config.raw_data.dataset_split.val
            ),
            self.download_dataset(
                dataset_name=self.dataset_config.raw_data.dataset_name,
                dataset_version=self.dataset_config.raw_data.dataset_version,
                dataset_split=self.dataset_config.raw_data.dataset_split.test
            )
        ]

        result = await asyncio.gather(*tasks)
        return {
            "train": result[0],
            "val": result[1],
            "test": result[2]
        }

    def __call__(self) -> Dict[str, Dataset]:
        return asyncio.run(self.download_datasets())


class DatasetPreprocessor:
    def __init__(self, tokenizer_path: str):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    def is_title(self, word: str, titles: list[str]) -> bool:
        """단어가 호칭인지 확인합니다."""
        return word.lower().rstrip('.') in titles

    def is_sentence_boundary(self, char: str, next_char: str, current_sentence: str, titles: list[str]) -> bool:
        """문장의 경계인지 판단합니다."""
        if char != '.' or next_char not in [' ', '\n']:
            return False

        words = current_sentence.strip().split()
        if not words:
            return False

        return not self.is_title(words[-1], titles)


    def split_sentences(self, text: str) -> list[str]:
        """텍스트를 문장 단위로 분리합니다."""
        titles = ['mr', 'mrs', 'ms', 'miss', 'dr', 'prof', 'rev', 'hon', 'st']

        result = []
        current = ""

        for i, char in enumerate(text):
            current += char

            if i + 1 < len(text) and self.is_sentence_boundary(char, text[i + 1], current, titles):
                result.append(current.strip())
                current = ""

        if current.strip():
            result.append(current.strip())

        return [s.replace("\n", " ") for s in result]

    def __call__(self, texts: list[Dict[str, Any]]) -> Generator[Dict[str, Any], None, None]:
        """Streaming 방식으로 텍스트를 처리하여 메모리 사용량을 줄입니다."""
        for text in tqdm.tqdm(texts["text"]):
            sentences = self.split_sentences(text)
            for sentence in sentences:
                yield {"text": sentence}


if __name__ == "__main__":
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
    
    # Streaming 방식으로 처리하고 파일에 저장
    total_count = 0
    datasets = downloader()
    
    for dataset_name, dataset in tqdm.tqdm(datasets.items(), desc="Processing datasets"):
        output_file = f"processed_{dataset_name}.jsonl"
        with open(output_file, 'w', encoding='utf-8') as f:
            for processed_item in preprocessor(dataset):
                f.write(json.dumps(processed_item, ensure_ascii=False) + '\n')
                total_count += 1
                # 메모리 사용량을 더 줄이기 위해 주기적으로 flush
                if total_count % 1000 == 0:
                    f.flush()
    
    print(f"Total processed items: {total_count}")
    print("Processed data saved to JSONL files")