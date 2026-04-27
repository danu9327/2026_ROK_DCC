"""
============================================================
🚀 시계열 예측 프로토타입 (40분 실습)
============================================================
항공 승객 데이터를 활용한 시계열 예측 대시보드

실행 방법:
    pip install streamlit pandas numpy matplotlib statsmodels scikit-learn
    streamlit run forecast_app.py

실습 목표:
    - Part 1(노트북)에서 배운 알고리즘을 하나의 앱으로 통합
    - 파라미터를 바꿔가며 예측 결과를 실시간으로 확인
    - 실제 프로토타입 개발 경험
============================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import mean_absolute_error, mean_squared_error
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')


# ============================================================
# 데이터 로드 함수
# ============================================================
@st.cache_data
def load_data():
    """항공 승객 데이터를 로드합니다."""
    data = sm.datasets.get_rdataset("AirPassengers").data
    data.columns = ['time', 'passengers']
    data['date'] = pd.date_range(start='1949-01', periods=len(data), freq='MS')
    data = data.set_index('date')
    return data['passengers']


# ============================================================
# 평가 함수
# ============================================================
def calc_metrics(actual, predicted):
    """MAE, RMSE, MAPE를 계산합니다."""
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    return {'MAE': round(mae, 2), 'RMSE': round(rmse, 2), 'MAPE': round(mape, 2)}


# ============================================================
# 예측 모델 함수들
# ============================================================
def seasonal_naive(train, steps):
    """Seasonal Naive: 작년 같은 달 값 사용"""
    last_year = train[-12:].values
    repeats = steps // 12 + 1
    forecast_values = np.tile(last_year, repeats)[:steps]
    return forecast_values


def holt_winters_forecast(train, steps, trend='mul', seasonal='mul'):
    """Holt-Winters 예측"""
    model = ExponentialSmoothing(
        train, trend=trend, seasonal=seasonal, seasonal_periods=12
    ).fit()
    return model.forecast(steps)


def sarima_forecast(train, steps, order=(1,1,1), seasonal_order=(1,1,1,12)):
    """SARIMA 예측"""
    model = SARIMAX(train, order=order, seasonal_order=seasonal_order).fit(disp=False)
    return model.forecast(steps)


# ============================================================
# 메인 앱
# ============================================================
def main():
    st.set_page_config(page_title="Time Series Forecasting", layout="wide")
    
    st.title("📈 시계열 예측 프로토타입")
    st.markdown("항공 승객 데이터를 활용한 시계열 예측 대시보드")
    st.markdown("---")

    # 데이터 로드
    ts = load_data()

    # ── 사이드바: 설정 ──────────────────────────────
    st.sidebar.header("⚙️ 설정")
    
    # 학습/테스트 분할
    split_year = st.sidebar.slider(
        "학습 데이터 마지막 연도", 
        min_value=1952, max_value=1959, value=1958
    )
    train = ts[:f'{split_year}']
    test = ts[f'{split_year + 1}':]
    forecast_steps = len(test)
    
    st.sidebar.markdown(f"""
    **데이터 분할**
    - Train: 1949 ~ {split_year} ({len(train)}개월)
    - Test: {split_year+1} ~ 1960 ({len(test)}개월)
    """)

    # 모델 선택
    st.sidebar.header("🔧 모델 선택")
    use_naive = st.sidebar.checkbox("Seasonal Naive", value=True)
    use_hw = st.sidebar.checkbox("Holt-Winters", value=True)
    use_sarima = st.sidebar.checkbox("SARIMA", value=True)

    # Holt-Winters 옵션
    if use_hw:
        st.sidebar.subheader("Holt-Winters 옵션")
        hw_trend = st.sidebar.selectbox("Trend", ['mul', 'add'], index=0)
        hw_seasonal = st.sidebar.selectbox("Seasonal", ['mul', 'add'], index=0)

    # SARIMA 옵션
    if use_sarima:
        st.sidebar.subheader("SARIMA 파라미터")
        col_p, col_d, col_q = st.sidebar.columns(3)
        p = col_p.number_input("p", 0, 3, 1)
        d = col_d.number_input("d", 0, 2, 1)
        q = col_q.number_input("q", 0, 3, 1)
        
        col_P, col_D, col_Q = st.sidebar.columns(3)
        P = col_P.number_input("P", 0, 2, 1)
        D = col_D.number_input("D", 0, 2, 1)
        Q = col_Q.number_input("Q", 0, 2, 1)

    # ── 탭 구성 ──────────────────────────────────
    tab1, tab2, tab3 = st.tabs(["📊 데이터 탐색", "🔮 예측 결과", "🔍 이상탐지"])

    # ── 탭 1: 데이터 탐색 ─────────────────────────
    with tab1:
        st.subheader("원본 데이터")
        
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(train, label='Train', color='steelblue')
        if len(test) > 0:
            ax.plot(test, label='Test', color='coral')
        ax.axvline(x=test.index[0], color='black', linestyle='--', alpha=0.3)
        ax.set_ylabel('Passengers (thousands)')
        ax.legend()
        ax.grid(alpha=0.3)
        st.pyplot(fig)
        plt.close()

        # 시계열 분해
        st.subheader("시계열 분해 (Multiplicative)")
        decomp = seasonal_decompose(ts, model='multiplicative', period=12)
        
        fig, axes = plt.subplots(4, 1, figsize=(12, 8))
        components = [
            (decomp.observed, 'Observed'),
            (decomp.trend, 'Trend'),
            (decomp.seasonal, 'Seasonal'),
            (decomp.resid, 'Residual')
        ]
        for ax, (comp, title) in zip(axes, components):
            ax.plot(comp, color='steelblue')
            ax.set_ylabel(title, fontsize=10)
            ax.grid(alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # 기본 통계
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("평균", f"{ts.mean():.0f}")
        col2.metric("표준편차", f"{ts.std():.0f}")
        col3.metric("최소", f"{ts.min():.0f}")
        col4.metric("최대", f"{ts.max():.0f}")

    # ── 탭 2: 예측 결과 ─────────────────────────
    with tab2:
        st.subheader("예측 결과 비교")
        
        if forecast_steps == 0:
            st.warning("테스트 데이터가 없습니다. 분할 연도를 조정하세요.")
        else:
            fig, ax = plt.subplots(figsize=(14, 6))
            ax.plot(train[-36:], label='Train (last 3 yrs)', color='steelblue', alpha=0.7)
            ax.plot(test, label='Actual', color='coral', linewidth=2.5)
            
            all_metrics = []

            # Seasonal Naive
            if use_naive:
                naive_vals = seasonal_naive(train, forecast_steps)
                naive_pred = pd.Series(naive_vals, index=test.index)
                ax.plot(naive_pred, label='Seasonal Naive', linestyle=':', alpha=0.8)
                m = calc_metrics(test, naive_pred)
                m['Model'] = 'Seasonal Naive'
                all_metrics.append(m)

            # Holt-Winters
            if use_hw:
                try:
                    hw_pred = holt_winters_forecast(
                        train, forecast_steps, trend=hw_trend, seasonal=hw_seasonal
                    )
                    ax.plot(hw_pred, label='Holt-Winters', linestyle='--', alpha=0.8)
                    m = calc_metrics(test, hw_pred)
                    m['Model'] = 'Holt-Winters'
                    all_metrics.append(m)
                except Exception as e:
                    st.error(f"Holt-Winters Error: {e}")

            # SARIMA
            if use_sarima:
                try:
                    with st.spinner("SARIMA 모델 학습 중..."):
                        sarima_pred = sarima_forecast(
                            train, forecast_steps,
                            order=(p, d, q),
                            seasonal_order=(P, D, Q, 12)
                        )
                    ax.plot(sarima_pred, label=f'SARIMA({p},{d},{q})({P},{D},{Q},12)', 
                            linestyle='--', alpha=0.8)
                    m = calc_metrics(test, sarima_pred)
                    m['Model'] = f'SARIMA({p},{d},{q})'
                    all_metrics.append(m)
                except Exception as e:
                    st.error(f"SARIMA Error: {e}")

            ax.legend(fontsize=10)
            ax.set_ylabel('Passengers (thousands)')
            ax.grid(alpha=0.3)
            ax.set_title('Forecast Comparison')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

            # 성능 테이블
            if all_metrics:
                st.subheader("📋 성능 비교")
                metrics_df = pd.DataFrame(all_metrics)[['Model', 'MAE', 'RMSE', 'MAPE']]
                
                # 최고 모델 하이라이트
                best_idx = metrics_df['MAPE'].idxmin()
                best_model = metrics_df.loc[best_idx, 'Model']
                
                st.dataframe(metrics_df.set_index('Model'), use_container_width=True)
                st.success(f"🏆 Best Model (MAPE 기준): **{best_model}** ({metrics_df.loc[best_idx, 'MAPE']}%)")
                
                # 지표 카드
                cols = st.columns(len(all_metrics))
                for i, m in enumerate(all_metrics):
                    with cols[i]:
                        st.metric(m['Model'], f"MAPE: {m['MAPE']}%")

    # ── 탭 3: 이상탐지 ─────────────────────────
    with tab3:
        st.subheader("이상탐지 (Residual 기반)")
        
        # 잔차 계산
        decomp = seasonal_decompose(ts, model='multiplicative', period=12)
        residual = decomp.resid.dropna()

        # 탐지 방법 선택
        method = st.radio("탐지 방법", ["Z-score", "IQR"], horizontal=True)
        
        if method == "Z-score":
            z_thresh = st.slider("Z-score 임계값", 1.5, 3.5, 2.5, 0.1)
            z_scores = (residual - residual.mean()) / residual.std()
            anomalies = residual[np.abs(z_scores) > z_thresh]
            upper_line = residual.mean() + z_thresh * residual.std()
            lower_line = residual.mean() - z_thresh * residual.std()
        else:
            iqr_mult = st.slider("IQR 배수", 1.0, 3.0, 1.5, 0.1)
            Q1, Q3 = residual.quantile(0.25), residual.quantile(0.75)
            IQR = Q3 - Q1
            upper_line = Q3 + iqr_mult * IQR
            lower_line = Q1 - iqr_mult * IQR
            anomalies = residual[(residual < lower_line) | (residual > upper_line)]

        # 시각화
        fig, ax = plt.subplots(figsize=(14, 5))
        ax.plot(residual, color='steelblue', alpha=0.7, label='Residual')
        ax.axhline(y=upper_line, color='red', linestyle='--', alpha=0.5, label='Threshold')
        ax.axhline(y=lower_line, color='red', linestyle='--', alpha=0.5)
        if len(anomalies) > 0:
            ax.scatter(anomalies.index, anomalies.values, color='red', 
                      s=100, zorder=5, label=f'Anomalies ({len(anomalies)})')
        ax.legend()
        ax.grid(alpha=0.3)
        ax.set_title(f'Anomaly Detection ({method})')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # 이상치 리스트
        if len(anomalies) > 0:
            st.subheader(f"🚨 탐지된 이상치: {len(anomalies)}개")
            anomaly_df = pd.DataFrame({
                'Date': anomalies.index.strftime('%Y-%m'),
                'Residual': anomalies.values.round(4),
                'Original': ts[anomalies.index].values
            })
            st.dataframe(anomaly_df.set_index('Date'), use_container_width=True)
        else:
            st.info("현재 설정에서 이상치가 탐지되지 않았습니다. 임계값을 낮춰보세요.")

    # ── 하단 정보 ───────────────────────────────
    st.markdown("---")
    st.markdown("""
    ### 💡 실습 가이드
    
    1. **사이드바**에서 학습/테스트 분할 비율을 바꿔보세요
    2. **SARIMA 파라미터** (p, d, q)를 조정하며 MAPE 변화를 관찰하세요
    3. **이상탐지** 탭에서 임계값을 조절하며 민감도 차이를 확인하세요
    4. Holt-Winters의 Trend/Seasonal을 `add`↔`mul`로 바꿔보세요
    """)


if __name__ == "__main__":
    main()
