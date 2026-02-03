import os
import requests
import pandas as pd

def collect_task(**context):
    try:
        # Airflow 컨테이너 환경변수에서 가져옴 (docker-compose에 작성)
        api_key = os.getenv("TMDB_API_KEY")
        base_url = os.getenv("TMDB_BASE_URL")
        
        save_dir = "/opt/airflow/data-prepare"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # Airflow 실행 날짜(ds)를 가져옴 (없으면 manual)
        ds = context.get("ds", "manual")

        def get_movie_details(movie_id):
            response = requests.get(
                f"{base_url}/movie/{movie_id}",
                params={"api_key": api_key}
            )
            response.raise_for_status()
            return response.json()

        print(f"데이터 수집 시작 (실행기준일: {ds})...")
        movies = []
        for page in range(1, 5+1):  # 5페이지 (100개 수집)
            response = requests.get(
                f"{base_url}/movie/popular",
                params={"api_key": api_key, "page": page}
            )
            response.raise_for_status()
            movies.extend(response.json()["results"])

        detailed_movies = []
        for movie in movies: 
            detail = get_movie_details(movie["id"])
            detailed_movies.append(detail)

        df = pd.DataFrame(detailed_movies)
        
        # 파일명: 날짜 포함
        file_path = os.path.join(save_dir, f"movies_{ds}.csv")
        df.to_csv(file_path, index=False, encoding='utf-8-sig') 
        
        print(f"성공: {len(detailed_movies)}개의 데이터를 가져왔습니다. 저장경로: {file_path}")
        return len(detailed_movies)

    except Exception as e:
        print(f"수집 중 오류 발생: {e}")
        raise e
