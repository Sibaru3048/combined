# NVIDIA CUDA 기반 Python 이미지 사용 (CUDA 11.8, Ubuntu 20.04)
FROM nvidia/cuda:11.8.0-runtime-ubuntu20.04

# 최신 패키지 설치 및 Python3 설치
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    curl \
    && apt-get clean

# 최신 pip 설치
RUN pip3 install --upgrade pip

# 작업 디렉터리 설정
WORKDIR /app

# requirements.txt 복사 및 설치
COPY ./requirements.txt /app/requirements.txt
RUN pip3 install --no-cache-dir -r /app/requirements.txt

# Hugging Face 모델 캐시 디렉터리 설정
ENV TRANSFORMERS_CACHE=/app/model_cache

# 디버깅 설정 추가
ENV CUDA_LAUNCH_BLOCKING=1

# 캐시 디렉터리 생성
RUN mkdir -p $TRANSFORMERS_CACHE

# Pegasus 모델 다운로드 및 캐싱
RUN python3 -c "\
from transformers import PegasusTokenizer, PegasusForConditionalGeneration;\
PegasusTokenizer.from_pretrained('EXP442/pegasus_summarizer', cache_dir='$TRANSFORMERS_CACHE');\
PegasusForConditionalGeneration.from_pretrained('EXP442/pegasus_summarizer', cache_dir='$TRANSFORMERS_CACHE')"

# NLLB 모델 다운로드 및 캐싱
RUN python3 -c "\
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM;\
AutoTokenizer.from_pretrained('EXP442/nllb_translator_pretrained', cache_dir='$TRANSFORMERS_CACHE');\
AutoModelForSeq2SeqLM.from_pretrained('EXP442/nllb_translator_pretrained', cache_dir='$TRANSFORMERS_CACHE')"

# 애플리케이션 코드 복사
COPY . /app

# 실행 파일 권한 설정 (선택 사항, Linux 권한 문제 방지)
RUN chmod -R 755 /app

# 포트 노출
EXPOSE 8080

# Uvicorn 서버 실행
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
