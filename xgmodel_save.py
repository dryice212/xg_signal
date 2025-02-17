import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
import joblib  # 모델 저장 및 로드에 필요한 라이브러리

# 데이터베이스 연결
db_path = "data/kospi200.db"
engine = create_engine(f'sqlite:///{db_path}')

# 데이터 로드
df = pd.read_sql('SELECT * FROM index_data', engine)

def add_indicators(df):
    """기술적 지표 추가 (CCI, RSI, MACD, ATR, EMA 등)"""
    df['TP'] = (df['high'] + df['low'] + df['close']) / 3
    df['CCI'] = (df['TP'] - df['TP'].rolling(14).mean()) / (0.015 * df['TP'].rolling(14).std())
    df['RSI'] = 100 - (100 / (1 + df['close'].pct_change().rolling(14).apply(lambda x: (x[x>0].sum() / -x[x<0].sum()) if x[x<0].sum() != 0 else np.nan)))
    df['EMA_12'] = df['close'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['ATR'] = df['high'] - df['low']
    df['OBV'] = (df['volume'] * np.sign(df['close'].diff())).cumsum()
    return df

df = add_indicators(df)

# 목표값 설정 (5일 후 수익률 기준으로 매수/매도 분류)
def define_labels(df, threshold=0.02):
    df['future_return'] = df['close'].pct_change(periods=5).shift(-5)
    df['label'] = 1  # 기본값을 1(중립)으로 설정
    df.loc[df['future_return'] > threshold, 'label'] = 2  # 매수 신호 (1 -> 2로 변경)
    df.loc[df['future_return'] < -threshold, 'label'] = 0  # 매도 신호 (-1 -> 0으로 변경)
    return df

df = define_labels(df)

# NaN 제거
df.dropna(inplace=True)

# 특징(Feature) 및 라벨 분리
features = ['CCI', 'RSI', 'MACD', 'MACD_signal', 'ATR', 'OBV']
X = df[features]
y = df['label']

# 클래스 불균형 해결 (SMOTE 적용)
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# 훈련/테스트 데이터 분리
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# XGBoost 모델 학습 (하이퍼파라미터 튜닝 적용)
model = XGBClassifier(
    objective='multi:softmax', num_class=3, eval_metric='mlogloss',
    max_depth=6, learning_rate=0.05, n_estimators=300, subsample=0.8
)
model.fit(X_train, y_train)

# 모델 저장
model_filename = 'xgboost_model.pkl'  # 저장할 파일 이름
joblib.dump(model, model_filename)
print(f"모델이 {model_filename} 파일에 저장되었습니다.")


# 예측 및 평가
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# 모델 예측 후 백테스트 실행
df['pred_signal'] = model.predict(X)
print("새로운 특징과 데이터 균형 적용 후 XGBoost 모델 평가 완료.")
