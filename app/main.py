from fastapi import FastAPI
import joblib
import pandas as pd
import os
import boto3  

app = FastAPI()

# 경로 설정
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")
MODEL_PATH = os.path.join(MODEL_DIR, "best_model.pkl")

def download_from_s3():
    """서버 시작 시 S3에서 최신 모델과 인코더를 다운로드합니다."""
    bucket_name = os.getenv("S3_BUCKET_NAME")
    if not bucket_name:
        print("경고: S3_BUCKET_NAME 환경변수가 설정되지 않았습니다.")
        return

    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    files = ["best_model.pkl", "main_genre_encoder.pkl", "original_language_encoder.pkl"]
    s3 = boto3.client('s3')

    print(f"--- S3({bucket_name})에서 모델 다운로드 시도 ---")
    try:
        for file_name in files:
            s3_path = f"models/latest/{file_name}"
            local_path = os.path.join(MODEL_DIR, file_name)
            
            # 다운로드 실행
            s3.download_file(bucket_name, s3_path, local_path)
            print(f"성공: {file_name} 다운로드 완료")
    except Exception as e:
        print(f"S3 다운로드 중 오류 발생: {e}")

# FastAPI 이벤트 핸들러: 서버가 시작될 때 자동으로 실행
@app.on_event("startup")
def startup_event():
    download_from_s3()

@app.get("/predict")
def predict(runtime: int, genre_encoded: int, lang_encoded: int, sentiment: float, overview_len: int):
    # 최신 베스트 모델 로드
    if not os.path.exists(MODEL_PATH):
        return {"error": "모델 파일이 없습니다. S3 설정을 확인하세요."}
    
    # 모델 로드
    model = joblib.load(MODEL_PATH)
    
    input_data = pd.DataFrame([[runtime, genre_encoded, lang_encoded, sentiment, overview_len]], 
                              columns=['runtime', 'genre_encoded', 'lang_encoded', 'overview_sentiment', 'overview_len'])
    
    prediction = model.predict(input_data)
    
    return {"predicted_rating": float(prediction[0])}