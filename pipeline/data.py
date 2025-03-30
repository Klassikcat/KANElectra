import re
from typing import List, Tuple
import tempfile
import os
import json
from pathlib import Path

from datasets import load_dataset, Dataset, concatenate_datasets
from flytekit import task, Resources
from flytekit.types.file import FlyteFile
from langcodes import Language, LanguageTagError
from transformers import AutoTokenizer

from .modules import split_sentences


@task(cache=True, cache_version="1.0", limits=Resources(cpu="2", mem="16Gi"))
def save_dataset_chunks(dataset: Dataset, chunk_size: int = 10000) -> List[FlyteFile]:
    """
    데이터셋을 청크 단위로 나누어 저장합니다.
    
    Args:
        dataset: 저장할 데이터셋
        chunk_size: 각 청크의 크기
    
    Returns:
        List[FlyteFile]: 청크 파일들의 리스트
    """
    chunk_files = []
    for i in range(0, len(dataset), chunk_size):
        chunk = dataset.select(range(i, min(i + chunk_size, len(dataset))))
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as tmp_file:
            chunk.to_json(tmp_file.name)
            chunk_files.append(FlyteFile(tmp_file.name))
    return chunk_files


@task(cache=True, cache_version="1.0", limits=Resources(cpu="4", mem="16Gi"))
def preprocess_data(data: Dataset, json_key: str = "text") -> List[str]:
    sentences = []
    for example in data:
        sentences.extend(split_sentences(example[json_key]))
    return sentences


@task(cache=True, cache_version="1.0", limits=Resources(cpu="2", mem="16Gi"))
def download_data(dataset_name: str, version: str, split: Tuple[int, int], language: str = "en") -> Dataset:
    """
    Download the Wikipedia dataset.
    
    Args:
        language (str): Language code to download ("en" or "ko")
    
    Returns:
        pl.DataFrame: Convert the downloaded dataset to a Polars DataFrame
    """
    if len(split) != 2:
        raise ValueError("The split must be a tuple of two integers")

    if not all(isinstance(i, int) and 0 <= i <= 100 for i in split):
        raise ValueError("The split must be a tuple of two integers between 0 and 100")
    
    if split[0] >= split[1]:
        raise ValueError("The first integer in the split must be less than the second integer")

    # Check if the version is in YYYYmmdd format
    if not re.match(r"^\d{4}\d{2}\d{2}$", version):
        raise ValueError(f"The version must be in YYYYmmdd format: {version}")

    try:
        # Validate language code
        Language.get(language)
        # Attempt to download only a specific percentile range of the dataset
        dataset = load_dataset(dataset_name, f"{version}.{language}", split=f'train[{split[0]}%:{split[1]}%]')
        # Convert the dataset to a Polars DataFrame
    except LanguageTagError as lang_error:
        raise ValueError(f"Invalid language code: {lang_error}")
    except Exception as e:
        raise ValueError(f"An error occurred while downloading the dataset: {e}")
    return dataset