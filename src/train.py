import os
import pandas as pd
import numpy as np
import wandb
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor

# 파라미터 파일 로드
import scripts.parameters as parameters

def train_task(**context):
    print("모델 학습 및 실험 기록 시작...")
    
    # 1. 경로 및 설정 준비
    save_dir = "/opt/airflow/data-prepare"
    ds = context.get("ds", "manual")
    input_file = os.path.join(save_dir, f"movies_cleaned_{ds}.csv")
    
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"전처리된 파일이 없습니다: {input_file}")

    # 2. Wandb 로그인 
    api_key = os.getenv("WANDB_API_KEY")
    if api_key:
        wandb.login(key=api_key)
    else:
        print("경고: WANDB_API_KEY가 설정되지 않았습니다. 오프라인 모드로 진행될 수 있습니다.")

    # 3. 실험 설정 로드
    cfg = parameters.MODEL_CONFIG
    m_type = cfg['model_type']
    p = cfg['params']
    exp_name = f"{m_type}_{ds}_depth{p.get('max_depth', 'N/A')}"

    wandb.init(
        project=parameters.WANDB_PROJECT,
        name=exp_name, 
        config=cfg,
        reinit=True
    )

    # 4. 데이터 로드 및 분리
    df = pd.read_csv(input_file)
    features = ['runtime', 'genre_encoded','lang_encoded','overview_sentiment','overview_len']  
    X = df[features]
    y = df['vote_average']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 5. 모델 빌드
    if m_type == "rf":
        # RF에 없는 파라미터들을 안전하게 제거
        p.pop("learning_rate", None) 
        p.pop("subsample", None)
        model = RandomForestRegressor(**p)
        print("Random Forest 모델 사용 (부적합 파라미터 제거 완료)")
        
    elif m_type == "xgboost":
        model = XGBRegressor(**p)
    elif m_type == "lgbm":
        model = LGBMRegressor(**p)

    # 6. 학습 및 평가
    model.fit(X_train, y_train)
    
    # Test RMSE
    preds = model.predict(X_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, preds))
    
    # Train RMSE (과적합 모니터링)
    train_preds = model.predict(X_train)
    train_rmse = np.sqrt(mean_squared_error(y_train, train_preds))
    rmse_gap = abs(test_rmse - train_rmse)

    # 7. 기록 및 저장
    wandb.log({
        "test_rmse": test_rmse,
        "train_rmse": train_rmse,
        "rmse_gap": rmse_gap
    })
    
    print(f"[{m_type}] 학습 완료! Test RMSE: {test_rmse:.4f}, Gap: {rmse_gap:.4f}")

    # 8. 모델 저장
    best_model_path = os.path.join(save_dir, "best_model.pkl")
    backup_model_path = os.path.join(save_dir, f"model_{ds}.pkl")

    # 8-1. 날짜별 백업
    joblib.dump(model, backup_model_path)
    print(f"현재 모델 백업 완료: {backup_model_path}")

    # 8-2. 기존 베스트 모델과 비교 (파일이 없으면 현재 모델이 베스트)
    is_best = False
    if not os.path.exists(best_model_path):
        is_best = True
    else:
        try:
            # 모델 파일만으로는 과거의 RMSE Gap을 알 수 없으므로 'score파일' 함께 관리
            score_path = os.path.join(save_dir, "best_score.txt")
            
            if os.path.exists(score_path):
                with open(score_path, "r") as f:
                    best_gap = float(f.read())
            else:
                best_gap = float('inf') # 파일 없으면 무한대

            if rmse_gap < best_gap:
                is_best = True
                print(f"새로운 베스트 모델 발견! (Gap: {best_gap:.4f} -> {rmse_gap:.4f})")
        except Exception as e:
            print(f"베스트 비교 중 오류 발생(무조건 갱신): {e}")
            is_best = True

    # 8-3. 베스트 모델일 경우 best_model.pkl 갱신
    if is_best:
        joblib.dump(model, best_model_path)
        with open(os.path.join(save_dir, "best_score.txt"), "w") as f:
            f.write(str(rmse_gap))
        print(f"최종 모델(best_model.pkl)이 갱신되었습니다.")
    else:
        print(f"현재 모델의 Gap({rmse_gap:.4f})이 기존보다 높아서 베스트모델을 갱신하지 않습니다.")

    wandb.finish()
