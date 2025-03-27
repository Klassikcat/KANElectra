import marimo

__generated_with = "0.11.29"
app = marimo.App(width="medium")


@app.cell
def _():
    import polars as pl
    from datasets import load_dataset
    return load_dataset, pl


@app.cell
def _(load_dataset):
    datasets_us = load_dataset("wikipedia", "20220301.en", split="train[:1%]")
    return (datasets_us,)


@app.cell
def _(datasets_us):
    datasets_us[0]
    return


@app.cell
def _(datasets_us):
    from transformers import AutoTokenizer

    text_sample = datasets_us[0]["text"]
    tokenizer = AutoTokenizer.from_pretrained("google/electra-base-discriminator")
    tokenized_text = tokenizer(datasets_us[0]["text"])

    print(len(tokenized_text.input_ids))
    return AutoTokenizer, text_sample, tokenized_text, tokenizer


@app.cell
def _():
    def is_title(word: str, titles: list[str]) -> bool:
        """단어가 호칭인지 확인합니다."""
        return word.lower().rstrip('.') in titles


    def is_sentence_boundary(char: str, next_char: str, current_sentence: str, titles: list[str]) -> bool:
        """문장의 경계인지 판단합니다."""
        if char != '.' or next_char not in [' ', '\n']:
            return False
        
        words = current_sentence.strip().split()
        if not words:
            return False
        
        return not is_title(words[-1], titles)


    def split_sentences(text: str) -> list[str]:
        """텍스트를 문장 단위로 분리합니다."""
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
    
    # 테스트
    text = "안녕하세요. Mr. Smith는 의사입니다.\n그의 메일은 mr.smith@gmail.com 입니다. Dr. Lee와 Prof. Kim이 왔어요. 그리고 Mrs. Park도 왔습니다."
    sentences = split_sentences(text)
    for sentence in sentences:
        print(sentence) 
    return (
        is_sentence_boundary,
        is_title,
        sentence,
        sentences,
        split_sentences,
        text,
    )


@app.cell
def _(split_sentences, text_sample, tokenizer):
    import tqdm
    from line_profiler import LineProfiler

    profiler = LineProfiler()

    splited_sentences = split_sentences(text_sample)

    def sample_func():
        total_texts = [tokenizer(i, max_length=512, return_attention_mask=True, return_token_type_ids=True, truncation=True, padding="max_length").input_ids for i in splited_sentences]
        return total_texts

    wrapped = profiler(sample_func)
    result = wrapped()
    profiler.print_stats()
    return (
        LineProfiler,
        profiler,
        result,
        sample_func,
        splited_sentences,
        tqdm,
        wrapped,
    )


@app.cell
def _(result):
    result
    return


@app.cell
def _(datasets_us):
    datasets_us
    return


if __name__ == "__main__":
    app.run()
