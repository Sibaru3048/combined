from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import PegasusTokenizer, PegasusForConditionalGeneration, AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import re

app = FastAPI()

# CORS 설정 추가
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 모든 출처 허용
    allow_credentials=True,
    allow_methods=["*"],  # 모든 HTTP 메서드 허용
    allow_headers=["*"],  # 모든 HTTP 헤더 허용
)

# 요청 데이터 모델 정의
class TextData(BaseModel):
    text: str

# 디바이스 설정 (GPU 사용 여부)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 요약 모델 및 토크나이저 설정
summarizer_model_name = "EXP442/pegasus_summarizer"
summarizer_tokenizer = PegasusTokenizer.from_pretrained(summarizer_model_name)
summarizer_model = PegasusForConditionalGeneration.from_pretrained(summarizer_model_name).to(device)

# 번역 모델 및 토크나이저 설정
translator_model_name = "EXP442/nllb_translator_pretrained"
translator_model = AutoModelForSeq2SeqLM.from_pretrained(
    translator_model_name, forced_bos_token_id=256098
).to(device)
translator_tokenizer = AutoTokenizer.from_pretrained(
    translator_model_name, src_lang='eng_Latn', tgt_lang='kor_Hang'
)

# 텍스트 분할 함수
def split_text_with_last_sentence_overlap(text, target_chunk_length=2048):
    sentences = re.split(r'(?<=[.!?]) +', text)
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= target_chunk_length:
            current_chunk += sentence + " "
        else:
            chunks.append(current_chunk.strip())
            current_chunk = chunks[-1].split()[-1] + " " + sentence + " "

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks 

# 긴 텍스트 요약 함수
def summarize_long_text(article_text, target_chunk_length=2048):
    chunks = split_text_with_last_sentence_overlap(article_text, target_chunk_length)
    summaries = []

    for chunk in chunks:
        inputs = summarizer_tokenizer(chunk, max_length=target_chunk_length, return_tensors="pt", truncation=True).to(device)
        summary_ids = summarizer_model.generate(inputs["input_ids"], max_length=100, min_length=50, length_penalty=2.0, num_beams=2, early_stopping=True)
        summary = summarizer_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        summaries.append(summary)

    return summaries 

# 텍스트 번역 함수
def translate_text(text):
    inputs = translator_tokenizer(text, return_tensors="pt", truncation=True).to(device)
    translated_ids = translator_model.generate(inputs["input_ids"], max_length=512, num_beams=4, early_stopping=True)
    translation = translator_tokenizer.decode(translated_ids[0], skip_special_tokens=True)
    return translation

# 요약과 번역을 결합하는 함수
def translate_and_combine_summaries(summaries):
    translated_summaries = [translate_text(summary) for summary in summaries]
    combined_translation = "\n".join(translated_summaries)
    return combined_translation 

# 엔드포인트 정의
@app.get("/")
async def root():
    return {"message": "Welcome to the VocaLabs API"}

@app.post("/summarize")
async def summarize_text(data: TextData):
    try:
        summaries = summarize_long_text(data.text)
        return {"summary": summaries}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error summarizing text: {str(e)}")

@app.post("/translate")
async def translate_text_endpoint(data: TextData):
    try:
        translation = translate_text(data.text)
        return {"translation": translation}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error translating text: {str(e)}")

@app.post("/process_text")
async def process_text(data: TextData):
    try:
        summaries = summarize_long_text(data.text)
        combined_translation = translate_and_combine_summaries(summaries)
        return {"summary_translation": combined_translation}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing text: {str(e)}")

# OPTIONS 요청 처리
@app.options("/process_text")
async def options_handler():
    return {"message": "OPTIONS request successful"}
