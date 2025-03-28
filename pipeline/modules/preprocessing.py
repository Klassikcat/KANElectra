
from typing import List
from transformers import AutoTokenizer


def is_title(word: str, titles: list[str]) -> bool:
    return word.lower().rstrip('.') in titles


def is_sentence_boundary(char: str, next_char: str, current_sentence: str, titles: list[str]) -> bool:
    if char != '.' or next_char not in [' ', '\n']:
        return False
        
    words = current_sentence.strip().split()
    if not words:
        return False
        
    return not is_title(words[-1], titles)


def split_sentences(text: str) -> list[str]:
    titles = ['mr', 'mrs', 'ms', 'miss', 'dr', 'prof', 'rev', 'hon']
    
    result = []
    current = ""
    
    for i, char in enumerate(text):
        current += char
        
        if i + 1 < len(text) and is_sentence_boundary(char, text[i + 1], current, titles):
            result.append(current.strip())
            current = ""
            
    if current.strip():
        result.append(current.strip())
        
    return result


def tokenize_sentences(sentences: List[str], tokenizer: AutoTokenizer, max_length: int = 512) -> List[List[int]]:
    return [
        tokenizer(
            sentence, 
            padding=True, 
            truncation=True,
            max_length=max_length,
            return_attention_mask=True,
            return_token_type_ids=True,
            return_tensors="pt",
        )["input_ids"] 
        for sentence in sentences
    ]