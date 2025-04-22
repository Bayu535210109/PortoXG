# === FILE: app.py ===
from flask import Flask, render_template, request
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from ta.trend import EMAIndicator, WMAIndicator, MACD
from ta.momentum import RSIIndicator, ROCIndicator
from ta.volatility import BollingerBands
from datetime import datetime, timedelta
import os

app = Flask(__name__)

tickers = ['NVDA', 'AAPL', 'GOOGL', '005930.KS', '9988.HK', '000660.KQ',
           'AMD', '0700.HK', 'TSLA', 'BABA', 'AMZN', 'INTC', 'MSFT']
sp500_ticker = '^GSPC'
start_date = (datetime.today() - timedelta(days=730)).strftime('%Y-%m-%d')
end_date = datetime.today().strftime('%Y-%m-%d')

def process_stock(ticker):
    try:
        df = yf.download(ticker, start=start_date, end=end_date)
        sp500 = yf.download(sp500_ticker, start=start_date, end=end_date)

        if df.empty or len(df) < 100:
            return None, None

        df = df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
        sp500 = sp500[['Close']].rename(columns={'Close': 'SP500_Close'}).dropna()
        df = df.join(sp500, how='inner')

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
        df['Target'] = df['Close'].shift(-1)
        df.dropna(inplace=True)

        features = [col for col in df.columns if col not in ['Target', 'Close']]
        X = df[features].copy()
        for col in X.columns:
            if X[col].dtype == 'object':
                X[col] = X[col].astype(float)
        y = df['Target'].astype(float).squeeze().values.ravel()

        model = XGBRegressor(n_estimators=100, random_state=42, verbosity=0)
        model.fit(X, y)

        pred_price = float(model.predict(X.iloc[[-1]]).item())
        last_close = df['Close'].iloc[-1]
        expected_return = (pred_price - last_close) / last_close

        return expected_return, last_close

    except Exception as e:
        print(f"[ERROR] {ticker}: {e}")
        return None, None

def prediksi_dan_alokasi(dana):
    hasil_return = {}
    for tkr in tickers:
        r, close = process_stock(tkr)
        try:
            if r is not None and float(r) > 0:
                hasil_return[tkr] = float(r)
        except Exception as e:
            print(f"[ERROR Filter Return] {tkr}: {e}")
            continue

    if not hasil_return:
        return {}

    total = sum(hasil_return.values())
    bobot = {tkr: r / total for tkr, r in hasil_return.items()}
    alokasi = {tkr: dana * b for tkr, b in bobot.items()}

    fig, ax = plt.subplots()
    ax.pie(alokasi.values(), labels=alokasi.keys(), autopct='%1.1f%%')
    ax.set_title("Alokasi Dana Prediksi")
    os.makedirs("static", exist_ok=True)
    plt.savefig("static/grafik_pie.png")
    plt.close()

    return alokasi

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    dana = None
    alokasi = None
    if request.method == 'POST':
        try:
            dana = float(request.form.get('dana_investasi'))
            alokasi = prediksi_dan_alokasi(dana)
        except Exception as e:
            return f"Terjadi error saat memproses: {e}"
        print("Dana:", dana)
        print("Alokasi:", alokasi)

    return render_template('prediction.html', dana=dana, alokasi=alokasi)

if __name__ == '__main__':
    app.run(debug=True)
