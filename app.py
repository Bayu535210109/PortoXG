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
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator, MACD
from ta.volatility import BollingerBands


app = Flask(__name__)

# ===== LOAD MODEL, SCALER, dan FEATURES SEKALI =====
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
features = joblib.load('models/features_global.pkl')

# List ticker sesuai model
tickers_13saham = ['NVDA', 'AAPL', 'GOOGL', '005930.KS', '9988.HK', '000660.KQ',
                   'AMD', '0700.HK', 'TSLA', 'ORCL', 'AMZN', 'INTC', 'MSFT']
tickers_sony = ['SONY', '1810.HK']

# Folder simpan grafik
if not os.path.exists('static'):
    os.makedirs('static')

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    if request.method == 'POST':
        dana = float(request.form.get('dana_investasi'))

        # --- Predict semua saham ---
        results = []
        
        # Prediksi saham dari model 13saham
        for ticker in tickers_13saham:
            harga_now, harga_pred = predict_ticker(ticker, "13saham")
            expected_return = (harga_pred - harga_now) / harga_now
            results.append({
                'ticker': ticker,
                'current_price': harga_now,
                'predicted_price': harga_pred,
                'expected_return': expected_return
            })
        
        # Prediksi saham dari model SONY_1810HK
        for ticker in tickers_sony:
            harga_now, harga_pred = predict_ticker(ticker, "SONY_1810HK")
            expected_return = (harga_pred - harga_now) / harga_now
            results.append({
                'ticker': ticker,
                'current_price': harga_now,
                'predicted_price': harga_pred,
                'expected_return': expected_return
            })

        # Convert ke DataFrame
        df_pred = pd.DataFrame(results)

        # --- Optimasi Portofolio dengan Gurobi ---
        model = Model("Portfolio Optimization")
        model.setParam('OutputFlag', 0)

        weights = {}
        for idx, row in df_pred.iterrows():
            weights[row['ticker']] = model.addVar(lb=0, ub=1, vtype=GRB.CONTINUOUS, name=row['ticker'])

        model.setObjective(sum(weights[t] * r for t, r in zip(df_pred['ticker'], df_pred['expected_return'])), GRB.MAXIMIZE)
        model.addConstr(sum(weights.values()) == 1)

        model.optimize()

        allocation = {t: weights[t].X for t in df_pred['ticker']}
        df_pred['allocation_percent'] = [allocation[t] for t in df_pred['ticker']]
        df_pred['allocation_nominal'] = df_pred['allocation_percent'] * dana

        # --- Generate Pie Chart ---
        plt.figure(figsize=(8, 8))
        plt.pie(df_pred['allocation_percent'], labels=df_pred['ticker'], autopct='%1.1f%%')
        plt.title('Distribusi Portofolio Saham')
        plt.tight_layout()
        pie_chart_path = os.path.join('static', 'portfolio_pie.png')
        plt.savefig(pie_chart_path)
        plt.close()

        # Format nominal IDR
        df_pred['allocation_nominal'] = df_pred['allocation_nominal'].apply(lambda x: f"IDR {x:,.0f}")
        df_pred['current_price'] = df_pred['current_price'].apply(lambda x: f"{x:.2f}")
        df_pred['predicted_price'] = df_pred['predicted_price'].apply(lambda x: f"{x:.2f}")
        df_pred['expected_return'] = df_pred['expected_return'].apply(lambda x: f"{x*100:.2f}%")
        df_pred = df_pred[['ticker', 'current_price', 'predicted_price', 'expected_return', 'allocation_nominal']]

        return render_template('prediction.html', tables=[df_pred.to_html(classes='table table-bordered')], pie_chart='static/portfolio_pie.png')

    return render_template('prediction.html')

def predict_ticker(ticker, model_key):
    model_info = models[model_key]
    model = model_info['model']
    scaler = model_info['scaler']

    df = yf.download(ticker, period='60d')
    if df.empty:
        return 0, 0  # atau handle error lebih baik

    df = df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
    df['Volume'] = np.log1p(df['Volume'])
    df['Lag1'] = df['Close'].shift(1)
    df['MA5'] = df['Close'].rolling(5).mean()
    df['MA10'] = df['Close'].rolling(10).mean()
    df['EMA10'] = EMAIndicator(close=df['Close'].squeeze(), window=10).ema_indicator()
    df['RSI'] = RSIIndicator(close=df['Close'].squeeze(), window=14).rsi()
    df['MACD'] = MACD(close=df['Close'].squeeze()).macd()
    df['BB_high'] = BollingerBands(close=df['Close'].squeeze()).bollinger_hband()
    df['BB_low'] = BollingerBands(close=df['Close'].squeeze()).bollinger_lband()
    df['Return'] = df['Close'].pct_change()
    df['RollingReturn5'] = df['Return'].rolling(5).mean()
    df['RollingReturn10'] = df['Return'].rolling(10).mean()
    df['Volatility5'] = df['Close'].pct_change().rolling(5).std()
    df['Price_Level'] = pd.qcut(df['Close'].squeeze(), q=4, labels=False)


    df.dropna(inplace=True)
    X_latest = df[features].iloc[-1:]

    X_scaled = scaler.transform(X_latest)
    pred_delta = model.predict(X_scaled)[0]
    last_close = df['Close'].iloc[-1]

    pred_price = last_close + pred_delta
    return last_close, pred_price

if __name__ == '__main__':
    app.run(debug=True)
