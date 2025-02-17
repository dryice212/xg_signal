import os
import pandas as pd
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
import mplcursors

# 데이터베이스 연결
DB_PATH = "data/kospi200.db"
engine = create_engine(f'sqlite:///{DB_PATH}')

# 지수 데이터 불러오기
df = pd.read_sql('SELECT date, close, buy_signal, sell_signal FROM index_data', engine)
df['date'] = pd.to_datetime(df['date'])


# 수익률 테이블 생성
trade_history = []
position = None  # 현재 포지션 (None, 'buy', 'sell')

for i, row in df.iterrows():
    if row['buy_signal'] == 1:
        position = 'buy'
        trade_history.append({'date': row['date'], 'buy_price': row['close'], 'sell_price': None, 'profit': None, 'cumulative_profit': None})

        # 이전 모든 sell_price 채우기
        for j in range(len(trade_history) - 1):
            if trade_history[j]['buy_price'] is None:
                trade_history[j]['buy_price'] = row['close']

    elif row['sell_signal'] == 1:
        position = 'sell'
        trade_history.append({'date': row['date'], 'buy_price': None, 'sell_price': row['close'], 'profit': None, 'cumulative_profit': None})

        # 이전 모든 buy_price 채우기
        for j in range(len(trade_history) - 1):
            if trade_history[j]['sell_price'] is None:
                trade_history[j]['sell_price'] = row['close']

# profit과 cumulative_profit 계산
for i in range(len(trade_history)):
    if trade_history[i]['buy_price'] is not None and trade_history[i]['sell_price'] is not None:
        trade_history[i]['profit'] = trade_history[i]['sell_price'] - trade_history[i]['buy_price']
    else:
        trade_history[i]['profit'] = None

cumulative_profit = 0
for i in range(len(trade_history)):
    if trade_history[i]['profit'] is not None:
        cumulative_profit += trade_history[i]['profit']
    trade_history[i]['cumulative_profit'] = cumulative_profit

# 데이터프레임 변환
trade_df = pd.DataFrame(trade_history)
trade_df.to_sql('returns_data', engine, if_exists='replace', index=False)

# 1. Figure와 Subplots 생성
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12), sharex=True)  # x축 공유

# 2. KOSPI 200 그래프 (ax1에 그림)
ax1.plot(df['date'], df['close'], label='KOSPI 200', color='blue')
ax1.scatter(df['date'][df['buy_signal'] == 1], df['close'][df['buy_signal'] == 1], marker='^', color='green', label='Buy Signal', alpha=1)
ax1.scatter(df['date'][df['sell_signal'] == 1], df['close'][df['sell_signal'] == 1], marker='v', color='red', label='Sell Signal', alpha=1)
ax1.set_title("KOSPI 200 with Buy/Sell Signals")
ax1.set_xlabel("Date")
ax1.set_ylabel("Index Price")
ax1.legend()
ax1.grid(True)  # Grid 추가

# 3. 누적 수익 그래프 (ax2에 그림)
ax2.plot(trade_df['date'], trade_df['cumulative_profit'], label='Cumulative Profit', color='purple')
ax2.set_title("Cumulative Profit Over Time")
ax2.set_xlabel("Date")
ax2.set_ylabel("Cumulative Profit")
ax2.legend()
ax2.grid(True)  # Grid 추가

# 4. 레이아웃 조정 (필수!)
plt.tight_layout() # 서브플롯 간 간격 자동 조정

# 5. x축 공유 설정 (매우 중요!)
plt.setp(ax1.get_xticklabels(), visible=False) # ax1의 x축 레이블 숨김

# 6. mplcursors 활성화 (필요한 경우)
mplcursors.cursor(ax1, hover=True)
mplcursors.cursor(ax2, hover=True)


# 7. 그래프 표시
plt.show()
