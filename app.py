from flask import Flask, render_template, request
import joblib
import pandas as pd
import numpy as np
import yfinance as yf
from gurobipy import Model, GRB
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import time
from ta.momentum import RSIIndicator, ROCIndicator
from ta.trend import MACD, EMAIndicator, WMAIndicator
from ta.volatility import BollingerBands
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# === Load Model dan Scaler ===
models = {
    "13saham": {
        "model": joblib.load('models/model_xgb_global_13saham.pkl'),
        "scaler": joblib.load('models/scaler_global_13saham.pkl')
    },
    "SONY_1810HK": {
        "model": joblib.load('models/model_xgb_global_SONY_1810HK.pkl'),
        "scaler": joblib.load('models/scaler_global_SONY_1810HK.pkl')
    }
}

# === Daftar Ticker ===
tickers_13saham = ['NVDA', 'AAPL', 'GOOGL', '005930.KS', '9988.HK', '000660.KQ',
                   'AMD', '0700.HK', 'TSLA', 'ORCL', 'AMZN', 'INTC', 'MSFT']
tickers_sony = ['SONY', '1810.HK']

sp500_ticker = '^GSPC'
start_date = '2023-01-01'
end_date = datetime.today().strftime('%Y-%m-%d')

# Buat folder static kalau belum ada
if not os.path.exists('static'):
    os.makedirs('static')

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    if request.method == 'POST':
        dana = float(request.form.get('dana_investasi'))
        results = []
        gagal_predict = []

        tickers_all = tickers_13saham + tickers_sony

        for ticker in tickers_all:
            model_key = "13saham" if ticker in tickers_13saham else "SONY_1810HK"
            harga_now, harga_pred = predict_ticker(ticker, model_key)

            if harga_now == 0 or harga_pred == 0:
                gagal_predict.append(ticker)
                continue

            expected_return = (harga_pred - harga_now) / harga_now
            results.append({
                'ticker': ticker,
                'current_price': harga_now,
                'predicted_price': harga_pred,
                'expected_return': expected_return
            })

        if not results:
            return render_template('prediction.html', error_message="Semua saham gagal diprediksi, coba lagi nanti.")

        df_pred = pd.DataFrame(results)

        # === Optimasi Portofolio ===
        model = Model("Portfolio Optimization")
        model.setParam('OutputFlag', 0)

        weights = {row['ticker']: model.addVar(lb=0, ub=1, vtype=GRB.CONTINUOUS) for idx, row in df_pred.iterrows()}
        model.setObjective(sum(weights[t] * r for t, r in zip(df_pred['ticker'], df_pred['expected_return'])), GRB.MAXIMIZE)
        model.addConstr(sum(weights.values()) == 1)
        model.optimize()

        df_pred['allocation_percent'] = [weights[t].X for t in df_pred['ticker']]
        df_pred['allocation_nominal'] = df_pred['allocation_percent'] * dana

        # === Pie Chart ===
        plt.figure(figsize=(8, 8))
        plt.pie(df_pred['allocation_percent'], labels=df_pred['ticker'], autopct='%1.1f%%')
        plt.title('Distribusi Portofolio Saham')
        plt.tight_layout()
        pie_chart_path = os.path.join('static', 'portfolio_pie.png')
        plt.savefig(pie_chart_path)
        plt.close()

        # === Format Tabel ===
        df_pred['allocation_nominal'] = df_pred['allocation_nominal'].apply(lambda x: f"IDR {x:,.0f}")
        df_pred['current_price'] = df_pred['current_price'].apply(lambda x: f"{x:.2f}")
        df_pred['predicted_price'] = df_pred['predicted_price'].apply(lambda x: f"{x:.2f}")
        df_pred['expected_return'] = df_pred['expected_return'].apply(lambda x: f"{x*100:.2f}%")
        df_pred = df_pred[['ticker', 'current_price', 'predicted_price', 'expected_return', 'allocation_nominal']]

        return render_template('prediction.html', 
                               tables=[df_pred.to_html(classes='table table-bordered', index=False)], 
                               pie_chart=pie_chart_path,
                               gagal_predict=gagal_predict)

    return render_template('prediction.html')

def predict_ticker(ticker, model_key):
    model_info = models[model_key]
    model = model_info['model']
    scaler = model_info['scaler']

    max_retry_days = 3
    success = False

    for retry in range(max_retry_days):
        current_end_date = (datetime.today() - timedelta(days=1 + retry)).strftime('%Y-%m-%d')
        df = yf.download(ticker, start=start_date, end=current_end_date, auto_adjust=False)
        sp500 = yf.download(sp500_ticker, start=start_date, end=current_end_date, auto_adjust=False)
        time.sleep(1.5)

        if not df.empty and not sp500.empty:
            success = True
            break

    if not success:
        return 0, 0

    try:
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
        sp500 = sp500[['Close']].rename(columns={'Close': 'SP500_Close'}).dropna()
        df = df.join(sp500, how='inner')
        df['Volume'] = np.log1p(df['Volume'])

        # === Generate Fitur ===
        if model_key == "13saham":
            df['Lag1'] = df['Close'].shift(1)
            df['Lag2'] = df['Close'].shift(2)
            df['Lag3'] = df['Close'].shift(3)
            df['Lag4'] = df['Close'].shift(4)
            df['MA5'] = df['Close'].rolling(5).mean()
            df['MA10'] = df['Close'].rolling(10).mean()
            df['MA20'] = df['Close'].rolling(20).mean()
            df['MA50'] = df['Close'].rolling(50).mean()
            df['EMA10'] = EMAIndicator(close=df['Close'], window=10).ema_indicator()
            df['EMA20'] = EMAIndicator(close=df['Close'], window=20).ema_indicator()
            df['EMA50'] = EMAIndicator(close=df['Close'], window=50).ema_indicator()
            df['WMA10'] = WMAIndicator(close=df['Close'], window=10).wma()
            df['WMA20'] = WMAIndicator(close=df['Close'], window=20).wma()
            df['Momentum'] = df['Close'] - df['Close'].shift(4)
            df['ROC'] = ROCIndicator(close=df['Close'], window=5).roc()
            df['Volatility'] = df['Close'].pct_change().rolling(5).std()
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
            df['RSI'] = RSIIndicator(close=df['Close'], window=14).rsi()
            macd = MACD(close=df['Close'])
            df['MACD'] = macd.macd()
            df['MACD_signal'] = macd.macd_signal()
            bb = BollingerBands(close=df['Close'])
            df['BB_high'] = bb.bollinger_hband()
            df['BB_low'] = bb.bollinger_lband()
            df['Year'] = df.index.year
            df['Month'] = df.index.month
            df['Day'] = df.index.day
            df['DayOfWeek'] = df.index.dayofweek
            df['Return'] = df['Close'].pct_change()
            df['Price_Level'] = pd.qcut(df['Close'], q=4, labels=False)

        elif model_key == "SONY_1810HK":
            df['Lag1'] = df['Close'].shift(1)
            df['MA5'] = df['Close'].rolling(5).mean()
            df['MA10'] = df['Close'].rolling(10).mean()
            df['EMA10'] = EMAIndicator(close=df['Close'], window=10).ema_indicator()
            df['RSI'] = RSIIndicator(close=df['Close'], window=14).rsi()
            df['MACD'] = MACD(close=df['Close']).macd()
            bb = BollingerBands(close=df['Close'])
            df['BB_high'] = bb.bollinger_hband()
            df['BB_low'] = bb.bollinger_lband()
            df['Return'] = df['Close'].pct_change()
            df['RollingReturn5'] = df['Return'].rolling(5).mean()
            df['RollingReturn10'] = df['Return'].rolling(10).mean()
            df['Volatility5'] = df['Close'].pct_change().rolling(5).std()
            df['Price_Level'] = pd.qcut(df['Close'], q=4, labels=False)

        df.dropna(inplace=True)

        if df.empty:
            return 0, 0

        features = [col for col in df.columns if col not in ['Target', 'Close']]
        X_latest = df[features].iloc[-1:]

        # Tambahkan print jumlah fitur
        print(f"{ticker} - Jumlah fitur: {X_latest.shape[1]} - Jumlah fitur scaler: {scaler.mean_.shape[0]}")


        # Cek jumlah fitur match
        if X_latest.shape[1] != scaler.mean_.shape[0]:
            return 0, 0

        X_scaled = scaler.transform(X_latest)
        pred_delta = model.predict(X_scaled)[0]
        last_close = df['Close'].iloc[-1]
        pred_price = last_close + pred_delta

        return last_close, pred_price

    except Exception:
        return 0, 0

if __name__ == '__main__':
    app.run(debug=True)
