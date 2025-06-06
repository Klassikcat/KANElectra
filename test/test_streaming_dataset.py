import os
import sys
from pathlib import Path
import pytest
from datasets import load_dataset
from transformers import AutoTokenizer
import pyarrow as pa
from pyarrow import parquet as pq
import tqdm
import torch

# 프로젝트 루트 디렉토리를 Python 경로에 추가
sys.path.append(str(Path(__file__).parent.parent))

from src.ElectraKAN.datamodule import StreamingElectraPretrainingDataset
from scripts.pretraining.data_processing import DatasetPreprocessor


def test_dataset_download_and_preprocess():
    """위키피디아 데이터셋 다운로드 및 전처리 테스트"""
    # 1. 데이터셋 다운로드
    dataset = load_dataset("wikipedia", "20220301.en", split="train[:1%]")
    assert len(dataset) > 0, "데이터셋이 비어있습니다."
    
    # 2. 전처리
    tokenizer = AutoTokenizer.from_pretrained("google/electra-base-discriminator")
    preprocessor = DatasetPreprocessor(tokenizer_path="google/electra-base-discriminator")
    processed_data = preprocessor(dataset)
    
    # 3. Parquet으로 저장
    output_path = "data/test_wikipedia.parquet"
    table = pa.Table.from_pylist(processed_data)
    pq.write_table(table, output_path)
    assert os.path.exists(output_path), "Parquet 파일이 생성되지 않았습니다."


def test_streaming_dataset_loading():
    """스트리밍 데이터셋 로딩 테스트"""
    # 1. 데이터셋 생성
    tokenizer = AutoTokenizer.from_pretrained("google/electra-base-discriminator")
    dataset = StreamingElectraPretrainingDataset(
        path="data/test_wikipedia.parquet",
        tokenizer=tokenizer,
        max_length=512,
        chunk_size=1000,
        text_column="text"
    )
    
    # 2. 데이터셋 크기 확인
    assert len(dataset) > 0, "데이터셋이 비어있습니다."
    
    # 3. 데이터 로딩 테스트
    for i in tqdm.tqdm(range(min(100, len(dataset))), desc="데이터 로딩 테스트"):
        # 데이터 로딩
        masked_input_ids, attention_mask, token_type_ids, input_ids = dataset[i]
        
        # 텐서 형태 확인
        assert masked_input_ids.shape == (512,), f"masked_input_ids shape error at index {i}"
        assert attention_mask.shape == (512,), f"attention_mask shape error at index {i}"
        assert token_type_ids.shape == (512,), f"token_type_ids shape error at index {i}"
        assert input_ids.shape == (512,), f"input_ids shape error at index {i}"
        
        # 데이터 타입 확인
        assert masked_input_ids.dtype == torch.long, f"masked_input_ids dtype error at index {i}"
        assert attention_mask.dtype == torch.long, f"attention_mask dtype error at index {i}"
        assert token_type_ids.dtype == torch.long, f"token_type_ids dtype error at index {i}"
        assert input_ids.dtype == torch.long, f"input_ids dtype error at index {i}"


def test_chunk_loading():
    """청크 로딩 테스트"""
    tokenizer = AutoTokenizer.from_pretrained("google/electra-base-discriminator")
    dataset = StreamingElectraPretrainingDataset(
        path="data/test_wikipedia.parquet",
        tokenizer=tokenizer,
        max_length=512,
        chunk_size=1000,
        text_column="text"
    )
    
    # 첫 번째 청크 로딩
    first_chunk = dataset.current_chunk
    assert len(first_chunk) > 0, "첫 번째 청크가 비어있습니다."
    
    # 두 번째 청크 로딩
    dataset._load_next_chunk()
    second_chunk = dataset.current_chunk
    assert len(second_chunk) > 0, "두 번째 청크가 비어있습니다."
    
    # 청크가 다른지 확인
    assert first_chunk != second_chunk, "청크가 중복되었습니다."


if __name__ == "__main__":
    # 테스트 실행
    test_dataset_download_and_preprocess()
    test_streaming_dataset_loading()
    test_chunk_loading()
    print("모든 테스트가 성공적으로 완료되었습니다!") 