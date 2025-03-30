import torch
from flytekit import FlyteFile
from typing import List, Iterator, Dict, Any, Optional
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
import tempfile
import fsspec as ff
import orjson
import logging
import re
import json


formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
handler = logging.StreamHandler()
handler.setFormatter(formatter)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.addHandler(handler)


class PretrainingDataset(Dataset):
    def __init__(self, file: FlyteFile, tokenizer: PreTrainedTokenizer, max_length: int) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.file = file
        self._index_file = None
        self._index = None
        self._build_index()

    def __len__(self) -> int:
        return len(self._index)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def _build_index(self) -> None:
        """
        JSON 파일에서 각 객체의 시작 위치를 인덱싱합니다.
        대용량 파일을 고려하여 청크 단위로 처리합니다.
        """
        self._index = []
        chunk_size = 1024 * 1024  # 1MB 청크 사이즈
        
        logger.info("Building index for JSON file...")
        
        with ff.open(self.file, 'rb') as f:
            # 첫 번째 문자는 '['
            next(f)
            
            buffer = b""
            position = 1  # '[' 이후 위치
            in_string = False  # 문자열 내부인지 추적
            escape_next = False  # 다음 문자를 이스케이프 처리할지 추적
            
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                    
                # 이전 청크의 마지막 부분과 현재 청크를 합침
                data = buffer + chunk
                
                i = 0
                while i < len(data):
                    byte = data[i]
                    
                    # 문자열 시작/종료 처리
                    if byte == ord('"') and not escape_next:
                        in_string = not in_string
                    # 중괄호 처리 (문자열 외부에서만)
                    elif not in_string and byte == ord('{') and not escape_next:
                        self._index.append(position + i)
                    
                    escape_next = False
                    i += 1
                
                # 마지막 청크의 일부를 다음 반복을 위해 버퍼에 저장
                buffer = data[-100:]  # 마지막 100바이트만 저장
                position += len(data) - len(buffer)
        
        logger.info(self._index)
        logger.info(f"Index built successfully. Found {len(self._index)} JSON objects.")

    def _get_json_at_index(self, idx: int) -> Optional[Dict[str, Any]]:
        """
        특정 인덱스의 JSON 객체를 반환합니다.
        
        Args:
            idx: 접근할 인덱스
            
        Returns:
            Optional[Dict[str, Any]]: JSON 객체 또는 None
        """
        if idx >= len(self._index):
            logger.info(f"Index {idx} out of range. Total indices: {len(self._index)}")
            return None
            
        logger.debug(f"Accessing JSON object at index {idx}")
        with ff.open(self.file, 'rb') as f:
            # 첫 번째 문자는 '['
            next(f)
            
            # 인덱스 위치로 이동
            f.seek(self._index[idx])
            
            # JSON 객체 읽기
            buffer = bytearray()
            in_object = True
            brace_count = 0  # 0으로 시작
            in_string = False  # 문자열 내부인지 추적
            escape_next = False  # 다음 문자를 이스케이프 처리할지 추적
            
            while in_object:
                chunk = f.read(1024)  # 1KB 청크로 읽기
                if not chunk:
                    logger.error(f"End of file reached while reading object at index {idx}")
                    break
                
                i = 0
                while i < len(chunk):
                    byte = chunk[i]
                    
                    # 이스케이프 문자 처리
                    if byte == ord('\\') and not escape_next:
                        escape_next = True
                        buffer.append(byte)
                    # 문자열 시작/종료 처리
                    elif byte == ord('"') and not escape_next:
                        in_string = not in_string
                        buffer.append(byte)
                    # 중괄호 처리 (문자열 외부에서만)
                    elif not in_string:
                        if byte == ord('{'):
                            brace_count += 1
                            buffer.append(byte)
                        elif byte == ord('}'):
                            brace_count -= 1
                            buffer.append(byte)
                            if brace_count == 0:
                                in_object = False
                                break
                        elif byte == ord(']'):
                            buffer.append(byte)
                            brace_count -= 1
                            if brace_count == 0:
                                in_object = False
                                break
                        else:
                            buffer.append(byte)
                    else:
                        buffer.append(byte)
                    
                    escape_next = False
                    i += 1
                
                if not in_object:
                    break
            
            try:
                # JSON 문자열로 디코딩
                json_str = buffer.decode('utf-8')
                # trailing comma 제거
                json_str = re.sub(r',\s*$', '', json_str)
                return orjson.loads(json_str)
            except Exception as e:
                logger.error(f"Error parsing JSON at index {idx}: {e}")
                return None

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        특정 인덱스의 데이터를 반환합니다.
        
        Args:
            idx: 접근할 인덱스
            
        Returns:
            Dict[str, Any]: 토큰화된 데이터
        """
        data = self._get_json_at_index(idx)
        if data is None:
            raise IndexError(f"Index {idx} out of range")
            
        text = data.get('text', '')
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_token_type_ids=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'token_type_ids': encoding['token_type_ids'].squeeze(),
            'masked_input_ids': self._mask_tokens(encoding['input_ids'].squeeze())
        }

    def _mask_tokens(self, input_ids: torch.Tensor) -> torch.Tensor:
        """RoBERTa style dynamic masking"""
        labels = input_ids.clone()
        probability_matrix = torch.full(labels.shape, 0.15)
        
        # 스페셜 토큰 마스킹 방지
        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True)
            for val in labels.tolist()
        ]
        probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
        
        # 마스킹할 위치 선택
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # 마스킹되지 않은 토큰은 -100으로 설정
        
        # 마스킹된 토큰의 80%는 [MASK]로 대체
        indices_mask = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        input_ids[indices_mask] = self.tokenizer.mask_token_id
        
        # 마스킹된 토큰의 10%는 랜덤 토큰으로 대체
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_mask
        input_ids[indices_random] = random_words[indices_random]
        
        # 나머지 10%는 원래 토큰 유지
        
        return input_ids

    def _stream_json_objects(self, file_obj) -> Iterator[str]:
        """
        JSON 배열을 스트리밍 방식으로 처리하여 개별 JSON 객체를 yield합니다.
        
        Args:
            file_obj: 파일 객체
            
        Yields:
            str: 개별 JSON 객체 문자열
        """
        buffer = ""
        in_object = False
        brace_count = 0
        
        for chunk in file_obj:
            chunk = chunk.decode('utf-8')
            for char in chunk:
                if char == '{':
                    if not in_object:
                        in_object = True
                    brace_count += 1
                    buffer += char
                elif char == '}':
                    brace_count -= 1
                    buffer += char
                    if brace_count == 0 and in_object:
                        in_object = False
                        yield buffer
                        buffer = ""
                else:
                    buffer += char

    @staticmethod
    def _merge_flytefiles(files: List[FlyteFile]) -> FlyteFile:
        """
        여러 FlyteFile을 하나로 합칩니다. 스트리밍 방식으로 처리합니다.
        
        Args:
            files: 합칠 FlyteFile들의 리스트
        
        Returns:
            FlyteFile: 합쳐진 파일
        """
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as output_file:
            with open(output_file.name, 'wb') as out_f:
                out_f.write(b'[')  # JSON 배열 시작
                for i, file in enumerate(files):
                    with ff.open(file, 'rb') as in_f:
                        # 첫 번째 파일이 아닌 경우 쉼표 추가
                        if i > 0:
                            out_f.write(b',')
                        
                        # 첫 번째 줄(열리는 대괄호) 건너뛰기
                        next(in_f)
                        
                        for line in in_f:
                            # 마지막 줄(닫는 대괄호) 제외
                            if line.strip().endswith(b']'):
                                continue
                            # 쉼표 제거
                            line = line.strip().rstrip(b',')
                            if line:
                                out_f.write(line + b'\n')
                
                out_f.write(b']')  # JSON 배열 종료
            return FlyteFile(output_file.name)

    @classmethod
    def load_from_flytefile(cls, files: List[FlyteFile], tokenizer: PreTrainedTokenizer, max_length: int):
        """
        FlyteFile을 로드하고 데이터셋을 반환합니다.
        
        Args:
            files: 로드할 FlyteFile 리스트
            tokenizer: 토크나이저
            max_length: 최대 시퀀스 길이
        """
        merged_file = cls._merge_flytefiles(files)
        return cls(merged_file, tokenizer, max_length)
        