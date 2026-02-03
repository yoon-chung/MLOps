import os
import pandas as pd
import numpy as np
import ast
import joblib
from sklearn.preprocessing import LabelEncoder
from textblob import TextBlob

# 헬퍼 함수: 감성 분석
def extract_sentiment(text):
    if pd.isna(text) or text == "":
        return 0.0
    return TextBlob(str(text)).sentiment.polarity

# 헬퍼 함수: 카테고리 인코딩 
def encode_categorical_col(df, col_name, save_dir):
    encoder_path = os.path.join(save_dir, f"{col_name}_encoder.pkl")
    if os.path.exists(encoder_path):
        print(f"{col_name} 기존 인코더 로드: {encoder_path}")
        le = joblib.load(encoder_path)
        df[col_name] = df[col_name].astype(str)
        valid_classes = set(le.classes_)
        df[f'{col_name}_encoded'] = df[col_name].map(
            lambda x: le.transform([x])[0] if x in valid_classes else -1
        )
    else:
        print(f"{col_name} 인코더 신규 생성 및 저장")
        le = LabelEncoder()
        df[f'{col_name}_encoded'] = le.fit_transform(df[col_name].astype(str))
        joblib.dump(le, encoder_path)
    return df

# Airflow가 호출할 메인 함수
def preprocess_task(**context):
    print("전처리 작업 시작...")
    
    save_dir = "/opt/airflow/data-prepare"
    
    # Airflow 실행 날짜(ds)를 받아 해당 파일 찾기
    ds = context.get("ds", "manual")
    input_filename = f"movies_{ds}.csv"
    output_filename = f"movies_cleaned_{ds}.csv"
    
    input_path = os.path.join(save_dir, input_filename)
    output_path = os.path.join(save_dir, output_filename)

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"수집된 파일이 없습니다: {input_path}")

    df = pd.read_csv(input_path)
    print(f"원본 데이터 로드 완료: {df.shape}")

    cols = ['id', 'title', 'original_language', 'budget', 'revenue', 'runtime', 'genres', 'release_date', 'vote_count', 'vote_average', 'overview']
    available_cols = [c for c in cols if c in df.columns]
    df = df[available_cols]
    
    df = df.dropna(subset=['runtime', 'release_date','budget', 'revenue'])
    df = df[df['vote_count'] > 5]

    df['release_year'] = pd.to_datetime(df['release_date']).dt.year

    def extract_genre_name(genre_str):
        try:
            genres = ast.literal_eval(genre_str)
            if isinstance(genres, list) and len(genres) > 0:
                return genres[0]['name']
            return 'Unknown'
        except:
            return 'Unknown'

    df['main_genre'] = df['genres'].apply(extract_genre_name)

    # 인코딩 실행
    df = encode_categorical_col(df, 'main_genre', save_dir)
    df = encode_categorical_col(df, 'original_language', save_dir)

    df.rename(columns={'main_genre_encoded': 'genre_encoded', 'original_language_encoded': 'lang_encoded'}, inplace=True)

    print("감성 분석 및 길이 추출 중...")
    df['overview_len'] = df['overview'].fillna('').apply(len)
    df['overview_sentiment'] = df['overview'].apply(extract_sentiment)

    # 불필요 컬럼 제거
    df_clean = df.drop(['genres', 'original_language', 'release_date','main_genre','overview'], axis=1)

    df_clean.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"전처리 완료! 저장 위치: {output_path}")
    
    return output_path # 다음 단계(Train)에서 파일명을 알 수 있도록 반환
