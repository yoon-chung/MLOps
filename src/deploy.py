import os
import boto3

def deploy_task(**context):
    """
    학습된 모델 및 인코더 파일을 S3 버킷으로 업로드합니다.
    """
    # 1. 환경 변수 로드 
    bucket_name = os.getenv("S3_BUCKET_NAME")
    access_key = os.getenv("AWS_ACCESS_KEY_ID")
    secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
    region = os.getenv("AWS_DEFAULT_REGION", "ap-northeast-2")
    
    # 2. 업로드할 파일 목록 정의 (모델 + 인코더 2종)
    source_dir = "/opt/airflow/data-prepare"
    files_to_upload = [
        "best_model.pkl",
        "main_genre_encoder.pkl",
        "original_language_encoder.pkl"
    ]
    
    # 3. S3 클라이언트 생성
    s3_client = boto3.client(
        's3',
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name=region
    )

    # 4. 파일별로 S3 업로드 실행
    print(f"S3 업로드 시작: 버킷명 - {bucket_name}")
    
    try:
        for file_name in files_to_upload:
            source_file = os.path.join(source_dir, file_name)
            # S3 내 저장 경로: models/latest/파일명
            s3_path = f"models/latest/{file_name}"
            
            print(f"업로드 중: {file_name} -> s3://{bucket_name}/{s3_path}")
            s3_client.upload_file(source_file, bucket_name, s3_path)
            
        print("모든 파일(모델 및 인코더)이 성공적으로 S3에 업로드되었습니다.")
        
    except Exception as e:
        print(f"S3 업로드 중 오류 발생: {e}")
        raise e

if __name__ == "__main__":
    deploy_task()
