from flytekit import task, Resources
from langcodes import Language, LanguageTagError
from datasets import load_dataset, Dataset
from .modules import split_sentences
from typing import List
import re


@task(cache=True, cache_version="1.0", limits=Resources(cpu="4", mem="16Gi"))
def preprocess_data(data: Dataset) -> List[str]:
    sentences = []
    for example in data:
        sentences.extend(split_sentences(example["text"]))
    return sentences


@task(cache=True, cache_version="1.0", limits=Resources(cpu="2", mem="16Gi"))
def download_data(dataset_name: str, version: str, language: str = "en") -> Dataset:
    """
    Download the Wikipedia dataset.
    
    Args:
        language (str): Language code to download ("en" or "ko")
    
    Returns:
        pl.DataFrame: Convert the downloaded dataset to a Polars DataFrame
    """

    # Check if the version is in YYYYmmdd format
    if not re.match(r"^\d{4}\d{2}\d{2}$", version):
        raise ValueError(f"The version must be in YYYYmmdd format: {version}")

    try:
        # Validate language code
        Language.get(language)
        # Attempt to download the dataset
        dataset = load_dataset(dataset_name, f"{version}.{language}")
        # Convert the dataset to a Polars DataFrame
    except LanguageTagError as lang_error:
        raise ValueError(f"Invalid language code: {lang_error}")
    except Exception as e:
        raise ValueError(f"An error occurred while downloading the dataset: {e}")
    return dataset

