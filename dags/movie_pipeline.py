from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import sys
import os

# Airflow 컨테이너 내에서 scripts 폴더를 인식할 수 있도록 경로 추가
sys.path.append('/opt/airflow')

# 함수들 불러오기
from scripts.collect import collect_task
from scripts.preprocess import preprocess_task
from scripts.train import train_task
from scripts.deploy import deploy_task

# 기본 설정
default_args = {
    'owner': 'Team1',
    'depends_on_past': False,
    'start_date': datetime(2026, 1, 23), # 오늘 날짜 기준
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# DAG 정의
with DAG(
    'tmdb_movie_rating_pipeline',
    default_args=default_args,
    description='TMDB 데이터 수집, 전처리, 모델 학습, 베스트모델 갱신까지의 파이프라인',
    schedule_interval=timedelta(days=1), # daily 실행
    catchup=False,
    tags=['mlops', 'movie'],
) as dag:

    # Task 1: 데이터 수집
    extract = PythonOperator(
        task_id='extract_tmdb_data',
        python_callable=collect_task,
        # collect_task 내부에서 **context를 통해 ds를 쓸 수 있게 함
        provide_context=True, 
    )

    # Task 2: 데이터 전처리
    transform = PythonOperator(
        task_id='preprocess_movie_data',
        python_callable=preprocess_task,
        provide_context=True,
    )

    # Task 3: 모델 학습 및 베스트 모델, 인코더 저장
    model_train = PythonOperator(
        task_id='train_and_save_best_model',
        python_callable=train_task,
        provide_context=True,
    )
    
    # Task 4: 베스트 모델/인코더 S3로 전송
    deploy = PythonOperator(
        task_id='deploy_model_to_s3',
        python_callable=deploy_task,
        provide_context=True,
    )
    
    # 실행 순서 정의 (수집 -> 전처리 -> 학습 -> S3 업로드)
    extract >> transform >> model_train >> deploy

