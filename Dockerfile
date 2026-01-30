FROM apache/airflow:2.7.1-python3.11

USER root
# 1. 시스템 패키지 설치 (libgomp1)
RUN apt-get update && \
    apt-get install -y --no-install-recommends libgomp1 && \
    apt-get autoremove -yqq --purge && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# 2. 다시 airflow 유저로 복귀 (보안 및 실행 권한 해결)
USER airflow

# 3. 파이썬 라이브러리 설치
COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt