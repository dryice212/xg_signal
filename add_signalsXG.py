import os
import pandas as pd
from sqlalchemy import create_engine
import joblib  # 모델 로딩을 위해 추가
import numpy as np  # NumPy 라이브러리 import

# 데이터베이스 파일 경로 설정
DB_FOLDER = "data"
DB_NAME = "kospi200.db"
DB_PATH = os.path.join(DB_FOLDER, DB_NAME)

# 모델 파일 경로 설정 (저장된 모델 파일 이름과 동일해야 함)
MODEL_FILENAME = 'xgboost_model.pkl'

# 데이터베이스 연결
engine = create_engine(f'sqlite:///{DB_PATH}')

def add_signals_to_db():
    """저장된 XGBoost 모델을 사용하여 매수/매도 신호를 생성하고 데이터베이스에 저장합니다."""

    # 모델 로드
    try:
        model = joblib.load(MODEL_FILENAME)
        print(f"모델 {MODEL_FILENAME} 로드 성공.")
    except FileNotFoundError:
        print(f"오류: 모델 파일 {MODEL_FILENAME}을 찾을 수 없습니다. 먼저 모델을 학습하고 저장해야 합니다.")
        return

    df = pd.read_sql('index_data', engine)

    # 기술적 지표 추가 (모델 학습에 사용된 것과 동일해야 함)
    df['TP'] = (df['high'] + df['low'] + df['close']) / 3
    df['CCI'] = (df['TP'] - df['TP'].rolling(14).mean()) / (0.015 * df['TP'].rolling(14).std())
    df['RSI'] = 100 - (100 / (1 + df['close'].pct_change().rolling(14).apply(lambda x: (x[x>0].sum() / -x[x<0].sum()) if x[x>0].sum() != 0 else np.nan)))
    df['EMA_12'] = df['close'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['ATR'] = df['high'] - df['low']
    df['OBV'] = (df['volume'] * np.sign(df['close'].diff())).cumsum()

    # NaN 값 제거 (모델 학습 전에 했던 것과 동일하게)
    df.dropna(inplace=True)

    # 예측에 사용할 특징 (모델 학습에 사용된 것과 동일해야 함)
    features = ['CCI', 'RSI', 'MACD', 'MACD_signal', 'ATR', 'OBV']
    X = df[features]

    # 모델 예측
    df['pred_signal'] = model.predict(X)

    # 매수/매도 신호 생성 (pred_signal 값을 기반으로)
    # 레이블 정의에 따라 0: 매도, 1: 중립, 2: 매수
    df['buy_signal'] = 0
    df['sell_signal'] = 0

    df.loc[df['pred_signal'] == 2, 'buy_signal'] = 1  # 모델이 '매수'라고 예측한 경우
    df.loc[df['pred_signal'] == 0, 'sell_signal'] = 1 # 모델이 '매도'라고 예측한 경우

    # 결과를 데이터베이스에 저장 (기존 테이블 덮어쓰기)
    df.to_sql('index_data', engine, if_exists='replace', index=False)
    print("모델 예측 기반 매수/매도 신호가 데이터베이스에 추가되었습니다.")

if __name__ == '__main__':
    add_signals_to_db()
