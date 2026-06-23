"""
미니 데이콘 챌린지 데이터 생성기 (한 번만 실행)

- 회귀: K-드라마 평균 시청률 예측
- 분류: 영화 흥행 등급 (망함/소박/중박/대박) 4단계

데이터는 합성이지만 도메인 친숙한 feature 이름을 사용합니다.
생성 규칙은 학습 가능하되 적당한 노이즈가 섞여 있어
랜덤 포레스트 정도면 70-80% 수준의 점수가 나오도록 설계했습니다.
"""

import numpy as np
import pandas as pd
from pathlib import Path

SEED = 42
np.random.seed(SEED)

BASE = Path(__file__).parent
REG_DIR = BASE / "data" / "regression"
CLF_DIR = BASE / "data" / "classification"
REG_DIR.mkdir(parents=True, exist_ok=True)
CLF_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 회귀: K-드라마 평균 시청률 예측
# ============================================================
def generate_kdrama_data(n=10000):
    genres   = ['로맨스', '사극', '스릴러', '코미디', '의학', '법정', '판타지']
    channels = ['지상파', '케이블', 'OTT']
    timeslots = ['평일심야', '평일저녁', '주말저녁', '주말심야']

    df = pd.DataFrame({
        'genre':              np.random.choice(genres, n),
        'lead_actor_tier':    np.random.randint(1, 6, n),       # 1=top급
        'writer_experience':  np.random.randint(0, 31, n),       # 작가 경력(년)
        'channel':            np.random.choice(channels, n, p=[0.45, 0.35, 0.20]),
        'episodes':           np.random.choice([8, 12, 16, 20, 24, 32], n,
                                               p=[0.10, 0.15, 0.40, 0.15, 0.15, 0.05]),
        'budget_billion_won': np.round(np.random.uniform(5, 100, n), 1),  # 총 제작비
        'release_quarter':    np.random.randint(1, 5, n),
        'timeslot':           np.random.choice(timeslots, n,
                                               p=[0.20, 0.40, 0.30, 0.10]),
        'is_remake':          np.random.choice([0, 1], n, p=[0.85, 0.15]),
        'lead_actor_age':     np.random.randint(20, 51, n),
    })

    # 타겟 생성: 학습 가능한 신호 + 적당한 노이즈
    rating = 4.0 \
        + (6 - df['lead_actor_tier']) * 0.6 \
        + np.minimum(df['writer_experience'] * 0.06, 2.0) \
        + df['channel'].map({'지상파': 1.2, '케이블': 0.0, 'OTT': -0.6}) \
        + df['genre'].map({'사극': 1.0, '의학': 0.8, '법정': 0.5,
                            '로맨스': 0.4, '스릴러': 0.3,
                            '코미디': 0.2, '판타지': 0.0}) \
        + df['timeslot'].map({'주말저녁': 1.5, '평일저녁': 0.7,
                               '주말심야': -0.3, '평일심야': -1.0}) \
        - df['is_remake'] * 0.4 \
        + np.sqrt(df['budget_billion_won']) * 0.22 \
        + np.where(df['episodes'] >= 24, -0.5, 0.0)  # 장편은 후반에 떨어지는 경향

    # 비선형 상호작용 한두 개 (feature engineering 보상)
    rating += np.where((df['genre'] == '로맨스') & (df['lead_actor_age'] < 30), 0.8, 0)
    rating += np.where((df['genre'] == '사극') & (df['channel'] == '지상파'), 0.7, 0)

    # 노이즈
    rating += np.random.normal(0, 1.6, n)
    rating = np.clip(rating, 0.3, 35.0).round(2)
    df['avg_rating'] = rating

    return df


def save_regression(df, out_dir):
    df = df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    df['id'] = np.arange(len(df))
    cols = ['id'] + [c for c in df.columns if c not in ['id', 'avg_rating']] + ['avg_rating']
    df = df[cols]

    split = 8000
    train = df.iloc[:split].copy()
    test_full = df.iloc[split:].copy()

    test_public = test_full.drop(columns=['avg_rating'])
    solution = test_full[['id', 'avg_rating']].copy()

    train.to_csv(out_dir / "train.csv", index=False)
    test_public.to_csv(out_dir / "test.csv", index=False)
    solution.to_csv(out_dir / "solution.csv", index=False)

    return train, test_public, solution


# ============================================================
# 분류: 영화 흥행 등급 (4클래스)
# ============================================================
def generate_movie_data(n=10000):
    genres = ['액션', '로맨스', '스릴러', '코미디', 'SF', '드라마', '호러', '애니메이션']
    ratings = ['전체관람가', '12세', '15세', '청불']

    df = pd.DataFrame({
        'genre':                np.random.choice(genres, n),
        'director_tier':        np.random.randint(1, 6, n),
        'lead_actor_tier':      np.random.randint(1, 6, n),
        'production_budget':    np.round(np.random.uniform(1, 300, n), 1),  # 억원
        'runtime_min':          np.random.randint(80, 181, n),
        'rating_age':           np.random.choice(ratings, n, p=[0.15, 0.30, 0.40, 0.15]),
        'release_month':        np.random.randint(1, 13, n),
        'is_sequel':            np.random.choice([0, 1], n, p=[0.80, 0.20]),
        'is_imported':          np.random.choice([0, 1], n, p=[0.60, 0.40]),
        'screen_count':         np.random.randint(100, 2500, n),
    })

    # 흥행 점수 (잠재 변수)
    score = 5.0 \
        + (6 - df['director_tier']) * 0.7 \
        + (6 - df['lead_actor_tier']) * 0.5 \
        + np.log1p(df['production_budget']) * 0.8 \
        + df['screen_count'] / 500 \
        + df['is_sequel'] * 1.0 \
        + df['genre'].map({'액션': 0.8, '애니메이션': 0.6, '드라마': 0.3,
                            'SF': 0.4, '코미디': 0.3, '로맨스': 0.0,
                            '스릴러': -0.1, '호러': -0.5}) \
        + df['rating_age'].map({'전체관람가': 0.5, '12세': 0.7,
                                 '15세': 0.3, '청불': -0.8}) \
        + df['release_month'].map(lambda m: 1.0 if m in [1, 2, 7, 8, 12] else 0.0)  # 성수기

    # 비선형 상호작용
    score += np.where((df['is_sequel'] == 1) & (df['genre'] == '액션'), 0.7, 0)
    score += np.where(df['runtime_min'] > 150, -0.3, 0)  # 너무 길면 -

    # 노이즈
    score += np.random.normal(0, 1.5, n)

    # 4분위로 등급 매기기 — 0=망함, 1=소박, 2=중박, 3=대박
    quantiles = np.quantile(score, [0.40, 0.70, 0.90])
    df['box_office_grade'] = np.digitize(score, quantiles)

    return df


def save_classification(df, out_dir):
    df = df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    df['id'] = np.arange(len(df))
    cols = ['id'] + [c for c in df.columns if c not in ['id', 'box_office_grade']] + ['box_office_grade']
    df = df[cols]

    split = 8000
    train = df.iloc[:split].copy()
    test_full = df.iloc[split:].copy()

    test_public = test_full.drop(columns=['box_office_grade'])
    solution = test_full[['id', 'box_office_grade']].copy()

    train.to_csv(out_dir / "train.csv", index=False)
    test_public.to_csv(out_dir / "test.csv", index=False)
    solution.to_csv(out_dir / "solution.csv", index=False)

    return train, test_public, solution


# ============================================================
# 실행
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("미니 데이콘 데이터 생성")
    print("=" * 60)

    # 회귀
    df_reg = generate_kdrama_data(10000)
    tr_r, te_r, sol_r = save_regression(df_reg, REG_DIR)
    print(f"\n[회귀] K-드라마 시청률 예측")
    print(f"  train: {len(tr_r)}행, test: {len(te_r)}행")
    print(f"  target 통계 — mean={df_reg['avg_rating'].mean():.2f}, "
          f"std={df_reg['avg_rating'].std():.2f}, "
          f"min={df_reg['avg_rating'].min():.2f}, "
          f"max={df_reg['avg_rating'].max():.2f}")
    print(f"  파일: {REG_DIR}")

    # 분류
    df_clf = generate_movie_data(10000)
    tr_c, te_c, sol_c = save_classification(df_clf, CLF_DIR)
    print(f"\n[분류] 영화 흥행 등급 분류")
    print(f"  train: {len(tr_c)}행, test: {len(te_c)}행")
    print(f"  클래스 분포 (전체):")
    counts = df_clf['box_office_grade'].value_counts().sort_index()
    label_names = {0: '망함', 1: '소박', 2: '중박', 3: '대박'}
    for k, v in counts.items():
        print(f"    {k} ({label_names[k]}): {v} ({v/len(df_clf)*100:.1f}%)")
    print(f"  파일: {CLF_DIR}")

    print("\n✅ 생성 완료")
