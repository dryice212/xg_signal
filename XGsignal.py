import os
import pandas as pd
import FinanceDataReader as fdr
from sqlalchemy import create_engine, Column, Integer, Date, REAL
from sqlalchemy.orm import declarative_base, sessionmaker
import joblib  # 모델 로딩을 위해 추가
import numpy as np  # NumPy 라이브러리 import
from datetime import datetime

# 데이터베이스 파일 경로 설정
DB_FOLDER = "data"
DB_NAME = "kospi200.db"
DB_PATH = os.path.join(DB_FOLDER, DB_NAME)

# 모델 파일 경로 설정 (저장된 모델 파일 이름과 동일해야 함)
MODEL_FILENAME = 'xgboost_model.pkl'

# 데이터베이스 연결
engine = create_engine(f'sqlite:///{DB_PATH}')
Session = sessionmaker(bind=engine)
session = Session()

# ORM 베이스 정의
Base = declarative_base()

# 지수 데이터 테이블 모델 정의 (매수·매도 신호 및 보류 상태 추가)
class IndexData(Base):
    __tablename__ = 'index_data'
    id = Column(Integer, primary_key=True, autoincrement=True)
    date = Column(Date, index=True)
    open = Column(REAL)
    high = Column(REAL)
    low = Column(REAL)
    close = Column(REAL)
    volume = Column(Integer)
    change = Column(REAL)
    buy_signal = Column(Integer, default=0)  # 매수 신호 (1: 매수, 0: 없음)
    sell_signal = Column(Integer, default=0)  # 매도 신호 (1: 매도, 0: 없음)
    pending_buy = Column(Integer, default=0)  # 매수 보류 상태 (1: 보류, 0: 없음)
    pending_sell = Column(Integer, default=0)  # 매도 보류 상태 (1: 보류, 0: 없음)

# 데이터베이스 생성 함수
def create_database():
    """
    데이터베이스와 테이블을 생성합니다.
    """
    Base.metadata.create_all(engine)
    print(f"'{DB_PATH}'에 데이터베이스와 테이블이 생성되었습니다.")

# KOSPI 200 데이터 가져오기 및 저장
def fetch_and_store_kospi200(start_date, end_date):
    df = fdr.DataReader('KS200', start_date, end_date)
    df.reset_index(inplace=True)
    df.rename(columns={'Date': 'date', 'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume', 'Change': 'change'}, inplace=True)
    df['date'] = pd.to_datetime(df['date'])

    # 매수·매도 신호 추가
    df['buy_signal'] = 0
    df['sell_signal'] = 0
    df['pending_buy'] = 0
    df['pending_sell'] = 0

    df.to_sql('index_data', engine, if_exists='replace', index=False)
    print("KOSPI 200 데이터가 'index_data' 테이블에 저장되었습니다.")


def generate_signals_and_print():
    """데이터 로드, 학습 모델 로드, 매수/매도 신호 생성 및 출력"""

    # 1. 데이터 로드
    try:
        df = pd.read_sql('index_data', engine)
        print("데이터 로드 성공.")
    except Exception as e:
        print(f"데이터 로드 실패: {e}")
        return

    # 2. 학습 모델 로드
    try:
        model = joblib.load(MODEL_FILENAME)
        print(f"모델 {MODEL_FILENAME} 로드 성공.")
    except FileNotFoundError:
        print(f"오류: 모델 파일 {MODEL_FILENAME}을 찾을 수 없습니다. 먼저 모델을 학습하고 저장해야 합니다.")
        return

    # 3. 기술적 지표 추가 (모델 학습에 사용된 것과 동일해야 함)
    df['TP'] = (df['high'] + df['low'] + df['close']) / 3
    df['CCI'] = (df['TP'] - df['TP'].rolling(14).mean()) / (0.015 * df['TP'].rolling(14).std())
    df['RSI'] = 100 - (100 / (1 + df['close'].pct_change().rolling(14).apply(lambda x: (x[x>0].sum() / -x[x<0].sum()) if x[x<0].sum() != 0 else np.nan)))
    df['EMA_12'] = df['close'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['ATR'] = df['high'] - df['low']
    df['OBV'] = (df['volume'] * np.sign(df['close'].diff())).cumsum()

    # 4. NaN 값 제거 (모델 학습 전에 했던 것과 동일하게)
    df.dropna(inplace=True)

    # 5. 예측에 사용할 특징 (모델 학습에 사용된 것과 동일해야 함)
    features = ['CCI', 'RSI', 'MACD', 'MACD_signal', 'ATR', 'OBV']
    X = df[features]

    # 6. 모델 예측
    df['pred_signal'] = model.predict(X)

    # 7. 매수/매도 신호 생성 (pred_signal 값을 기반으로)
    # 레이블 정의에 따라 0: 매도, 1: 중립, 2: 매수
    df['buy_signal'] = 0
    df['sell_signal'] = 0

    df.loc[df['pred_signal'] == 2, 'buy_signal'] = 1  # 모델이 '매수'라고 예측한 경우
    df.loc[df['pred_signal'] == 0, 'sell_signal'] = 1 # 모델이 '매도'라고 예측한 경우

    # 8. 오늘 날짜의 신호 출력
    today = datetime.today().strftime('%Y-%m-%d')
    today_data = df[df['date'] == today]

    if not today_data.empty:
        buy_signal = today_data['buy_signal'].values[0]
        sell_signal = today_data['sell_signal'].values[0]

        print(f"\n오늘 ({today})의 신호:")
        print(f"매수 신호: {buy_signal}")
        print(f"매도 신호: {sell_signal}")
    else:
        print(f"오늘 ({today}) 데이터가 없습니다.")


if __name__ == '__main__':
    start_date = '2010-01-02'
    end_date = datetime.today().strftime('%Y-%m-%d')

    # 1. 데이터베이스 생성 (최초 실행 시에만 필요)
    create_database()

    # 2. KOSPI 200 데이터 가져오기 및 저장
    fetch_and_store_kospi200(start_date, end_date)

    # 3. 신호 생성 및 출력
    generate_signals_and_print()
