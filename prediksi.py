import yfinance as yf
import pandas as pd
import numpy as np
import joblib
from ta.momentum import RSIIndicator, ROCIndicator
from ta.trend import MACD, EMAIndicator, WMAIndicator
from ta.volatility import BollingerBands
from datetime import datetime
from sklearn.preprocessing import StandardScaler

# --- Load model, scaler, dan fitur global ---
model = joblib.load("models/model_xgb_global_13saham.pkl")
scaler = joblib.load("models/scaler_global_13saham.pkl")
features = joblib.load("models/features_global.pkl")

# --- Load data terbaru untuk 1 saham ---
ticker = "NVDA"
sp500_ticker = '^GSPC'
start_date = (datetime.today() - pd.Timedelta(days=730)).strftime('%Y-%m-%d')
end_date = datetime.today().strftime('%Y-%m-%d')
df = yf.download(ticker, start=start_date, end=end_date)
sp500 = yf.download(sp500_ticker, start=start_date, end=end_date)
df = df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
sp500 = sp500[['Close']].rename(columns={'Close': 'SP500_Close'}).dropna()
df = df.join(sp500, how='inner')

# === FITUR TEKNIKAL & INTERAKSI ===
df['Volume'] = np.log1p(df['Volume'])
df['Lag1'] = df['Close'].shift(1)
df['Lag2'] = df['Close'].shift(2)
df['Lag3'] = df['Close'].shift(3)
df['Lag4'] = df['Close'].shift(4)
df['MA5'] = df['Close'].rolling(5).mean()
df['MA10'] = df['Close'].rolling(10).mean()
df['MA20'] = df['Close'].rolling(20).mean()
df['MA50'] = df['Close'].rolling(50).mean()
df['EMA10'] = EMAIndicator(close=df['Close'].squeeze(), window=10).ema_indicator()
df['EMA20'] = EMAIndicator(close=df['Close'].squeeze(), window=20).ema_indicator()
df['EMA50'] = EMAIndicator(close=df['Close'].squeeze(), window=50).ema_indicator()
df['WMA10'] = WMAIndicator(close=df['Close'].squeeze(), window=10).wma()
df['WMA20'] = WMAIndicator(close=df['Close'].squeeze(), window=20).wma()
df['Momentum'] = df['Close'] - df['Close'].shift(4)
df['ROC'] = ROCIndicator(close=df['Close'].squeeze(), window=5).roc()
df['Volatility'] = df['Close'].pct_change().rolling(window=5).std()
df['High_Low'] = df['High'] - df['Low']
df['Close_Open'] = df['Close'] - df['Open']
df['Rolling_Max_5'] = df['Close'].rolling(5).max()
df['Rolling_Min_5'] = df['Close'].rolling(5).min()
df['Price_Change'] = df['Close'].diff()
df['3D_Trend'] = df['Close'] - df['Close'].shift(3)
df['Lag1_ratio'] = df['Lag1'] / df['Lag2']
df['Lag2_ratio'] = df['Lag3'] / df['Lag4']
df['MA5_vs_EMA10'] = df['MA5'] - df['EMA10']
df['MA5_to_MA20'] = df['MA5'] / df['MA20']
df['EMA10_to_EMA50'] = df['EMA10'] / df['EMA50']
df['WMA10_to_WMA20'] = df['WMA10'] / df['WMA20']
df['RSI'] = RSIIndicator(close=df['Close'].squeeze(), window=14).rsi()
macd = MACD(close=df['Close'].squeeze())
df['MACD'] = macd.macd()
df['MACD_signal'] = macd.macd_signal()
bb = BollingerBands(close=df['Close'].squeeze())
df['BB_high'] = bb.bollinger_hband()
df['BB_low'] = bb.bollinger_lband()
df['Year'] = df.index.year
df['Month'] = df.index.month
df['Day'] = df.index.day
df['DayOfWeek'] = df.index.dayofweek
df['Return'] = df['Close'].pct_change()

df.dropna(inplace=True)

# --- Ambil data terakhir untuk prediksi ---
X = df[features]
X_latest = X.iloc[-1:]
X_scaled = scaler.transform(X_latest)

# --- Prediksi ---
y_pred = model.predict(X_scaled)[0]
print(f"\n📈 Prediksi harga {ticker} untuk hari berikutnya: {y_pred:.2f}")
