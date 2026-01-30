# 실험용 파라미터 설정
WANDB_PROJECT = "movie-rating-predictor"
SAVE_DIR = "/opt/airflow/data-prepare"

MODEL_CONFIG = {
    "model_type": "xgboost",  # "rf", "xgboost", "lgbm" 중 선택
    "params": {
        "n_estimators": 200,      # 나무의 개수
       "learning_rate": 0.05,    # 학습률  # rf일 때 null
        "max_depth": 4,           # 나무 깊이, 너무 크면 과적합 발생
        "subsample": 0.7,         # 데이터 샘플링 비율 # rf일 때 null
        "random_state": 42,
      #  "force_col_wise":True,    # 메모리 용량이 충분하지 않을 때 메모리 효율을 높이는 파라미터(Label을 feature로 넣으면 발생)
    }
}