# 🏆 7회차 미니 데이콘 챌린지 — 강사 가이드

## 📦 파일 구성

```
mini_dacon/
├── server.py                   # FastAPI 챌린지 서버
├── generate_data.py            # 데이터 생성
├── student_notebook.ipynb      # 배포용 Colab 노트북
├── data/
│   ├── regression/
│   │   ├── train.csv           
│   │   ├── test.csv            
│   │   └── solution.csv        
│   └── classification/
│       ├── train.csv
│       ├── test.csv
│       └── solution.csv        
├── leaderboard.db              # SQLite (서버 실행시 자동 생성)
└── README.md                   
```

## 🎯 챌린지 사양

### 회귀: K-드라마 평균 시청률 예측
- **타겟**: `avg_rating` (실수, 단위 %)
- **평가**: RMSE (낮을수록 좋음)
- **train**: 8000행, **test**: 2000행
- **베이스라인 (RF 기본)**: RMSE ≈ 1.69
- **이론적 하한**: RMSE ≈ 1.6 (노이즈 std)

### 분류: 영화 흥행 등급 (4클래스)
- **타겟**: `box_office_grade` (0=망함, 1=소박, 2=중박, 3=대박)
- **평가**: Accuracy
- **train**: 8000행, **test**: 2000행
- **클래스 분포**: 40 / 30 / 20 / 10 (불균형)
- **베이스라인 (RF 기본)**: Acc ≈ 55%
- **상위권 도달선**: Acc ≈ 60-65%

## 🚀 수업 당일 운영 절차

### Step 1. 사전 준비

```bash
# 서버 PC에서
pip install fastapi uvicorn pandas scikit-learn pydantic python-multipart

cd ~/path/to/mini_dacon
python generate_data.py     
```

### Step 2. 서버 띄우기 (터미널 A)

```bash
cd ~/path/to/mini_dacon
python server.py
```

`Uvicorn running on http://0.0.0.0:9000` 

### Step 3. 공개 URL 발급 (터미널 B)

```bash
cloudflared tunnel --url http://localhost:9000
```

`https://concerns-kenneth-plastic-thesaurus.trycloudflare.com` 

### Step 4. 리더보드 화면 띄우기 (강사 PC, 빔/대형 모니터)

브라우저로 `https://*.trycloudflare.com` 