import streamlit as st   # 로컬환경: streamlit run app.py 로 실행
import pandas as pd
import joblib
import os
from textblob import TextBlob
from deep_translator import GoogleTranslator # 번역 라이브러리 추가

# 1. 파일 경로 설정
SAVE_DIR = "./app/model"  # 컨테이너 내부 마운트 경로
MODEL_PATH = os.path.join(SAVE_DIR, "best_model.pkl")
GENRE_ENC_PATH = os.path.join(SAVE_DIR, "main_genre_encoder.pkl")
LANG_ENC_PATH = os.path.join(SAVE_DIR, "original_language_encoder.pkl")

# 2. 모델 및 인코더 로드 함수 (캐싱 처리로 속도 향상)
@st.cache_resource
def load_assets():
    model = joblib.load(MODEL_PATH)
    le_genre = joblib.load(GENRE_ENC_PATH)
    le_lang = joblib.load(LANG_ENC_PATH)
    return model, le_genre, le_lang

# 3. 메인 UI 구성
translator = GoogleTranslator()
st.set_page_config(page_title="영화 평점 예측 서비스 (다국어 지원)", page_icon="🎬")
st.title("🎬 AI 영화 평점 예측기 (다국어 지원)")
st.markdown("""
입력하신 영화 정보를 바탕으로 약 70-80% 정확도로 예상 평점을 분석합니다.
\n줄거리의 분위기(감성)까지 점수에 반영됩니다!
""")

try:
    model, le_genre, le_lang = load_assets()

    with st.form("movie_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            runtime = st.number_input("상영시간(분)", min_value=1, max_value=300, value=120)
            genre = st.selectbox("주요 장르", options=le_genre.classes_)
            
        with col2:
            language = st.selectbox("언어", options=le_lang.classes_)
            
        overview = st.text_area("영화 줄거리", 
                                placeholder="이곳에 영화의 줄거리를 입력하세요. 길이에 따라 예측값이 달라집니다.\n한글로 입력하셔도 AI가 번역하여 분석합니다.")
        
        submit = st.form_submit_button("예상 평점 확인")

    if submit:
        # --- 데이터 전처리 (Inference) ---
        with st.spinner('AI가 줄거리를 분석 중입니다...'):
            # --- [핵심] 번역 파이프라인 ---
            try:
                # 한국어 등 비영어권을 위해 영문으로 번역
                translated = GoogleTranslator(source='auto', target='en').translate(overview)                
            except:
                translated = overview # 번역 실패 시 원문 사용

            # 1. 텍스트 특징 추출
            blob = TextBlob(translated)
            sentiment = blob.sentiment.polarity
            overview_len = len(translated)
        
            # 2. 인코딩 처리
            genre_encoded = le_genre.transform([genre])[0]
            lang_encoded = le_lang.transform([language])[0]
        
            # 3. 모델 입력 데이터 생성 (features 순서 중요)
            input_data = pd.DataFrame([[
                runtime, 
                genre_encoded, 
                lang_encoded, 
                sentiment,
                overview_len,]], 
                columns=['runtime', 'genre_encoded','lang_encoded','overview_sentiment','overview_len'])
        
            # 4. 예측 실행
            prediction = model.predict(input_data)[0]
        
            # --- 결과 출력 ---
            st.divider()
            st.subheader("📊 분석 결과")
        
            result_col1, result_col2 = st.columns(2)
            result_col1.metric("예상 평점", f"{prediction:.2f} / 10")
        
        # 감성 점수에 따른 라벨링 
            if sentiment > 0.1:
                label = "희망적/긍정적 😊"
            elif sentiment < -0.1:
                label = "어둡고/부정적 🌑"
            else:
                label = "중립적 😐"
            
            result_col2.markdown(f"**줄거리 분위기**")
            result_col2.write(f"{label} (점수: {sentiment:.2f})")
        
        st.info(f"💡 이 시뮬레이션은 '{genre}' 장르와 '{language}' 언어의 특성을 반영한 결과입니다.")

except FileNotFoundError:
    st.error("모델이나 인코더 파일을 찾을 수 없습니다. 먼저 전처리와 학습을 완료해 주세요!")