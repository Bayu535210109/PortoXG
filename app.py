from flask import Flask, render_template, request, session, send_file
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
from xgboost import XGBRegressor
from ta.momentum import RSIIndicator, ROCIndicator
from ta.trend import MACD, EMAIndicator, WMAIndicator
from ta.volatility import BollingerBands
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

app = Flask(__name__)
app.secret_key = 'porto_xg_secret_key_2025'

def load_model_by_ticker(ticker):
    model_path = f"models/model_{ticker}.pkl"
    scaler_path = f"models/scaler_{ticker}.pkl"
    features_path = f"models/features_{ticker}.pkl"

    if not all(os.path.exists(p) for p in [model_path, scaler_path, features_path]):
        print(f"[ERROR] Model files not found for {ticker}")
        return 0, 0, 0, pd.DataFrame(), []


    try:
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        features = joblib.load(features_path)
    except Exception as e:
        print(f"[ERROR] Gagal memuat model {ticker}: {e}")
        return None, None, None

    return model, scaler, features

# === Daftar Ticker ===
tickers_13saham = ['NVDA', 'AAPL', 'GOOGL', '005930.KS', '000660.KQ',
                   'AMD', 'TSLA', 'ORCL', 'AMZN', 'INTC', 'MSFT','9988.HK', '0700.HK','GOTO.JK']
tickers_xiaomi = ['1810.HK']
tickers_sony = ['SONY']
sp500_ticker = '^GSPC'
start_date = '2023-01-01'
end_date = datetime.today().strftime('%Y-%m-%d')

@app.route('/')
def home():
    return render_template('home.html')

def evaluate_portfolio(df):
    expected_return = (df['allocation_percent'] * df['expected_return']).sum()
    realized_return = ((df['predicted_price_3mo'] - df['current_price']) / df['current_price'] * df['allocation_percent']).sum()
    mape = np.mean(np.abs((df['predicted_price_3mo'] - df['current_price']) / df['current_price']))
    return {
        'expected_return': expected_return,
        'realized_return': realized_return,
        'mape': mape
    }

@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    if request.method == 'POST':
        dana_input_raw = request.form.get('dana_investasi')
        dana_input = float(dana_input_raw) if dana_input_raw else 0
        selected_ticker = request.form.get('selected_ticker')

        # ✅ Jika hanya memilih saham (tanpa submit dana baru)
        if selected_ticker and 'last_results' in session and 'last_dana' in session:
            df_alloc = optimize_portfolio(session['last_results'], session['last_dana'])
            pie_chart_path = generate_pie_chart(df_alloc)
            df_view = format_df_for_display(df_alloc)

            evaluasi = evaluate_portfolio(pd.DataFrame(session['last_results']))

            chart_path = f"static/chart_13saham/prediksi_chart_{selected_ticker}.png"
            if not os.path.exists(chart_path):
                chart_path = None

            context = {
                'tables': [df_view.to_html(classes='table table-bordered', index=False, escape=False)],
                'pie_chart': pie_chart_path,
                'tickers': [r['ticker'] for r in session['last_results']],
                'gagal_predict': [],
                'dana': session['last_dana'],
                'selected_ticker': selected_ticker,
                'excluded_tickers': [],
                'chart_path': chart_path,
                'chart_paths': chart_path,
                'chart_ticker': selected_ticker if chart_path else None,
                'evaluasi': evaluasi
            }
            return render_template('prediction.html', **context)
        
        dana = dana_input
        results = []
        gagal = []

        for ticker in tickers_13saham:
            harga_now, harga_pred, harga_pred_3bulan, df, future_preds = predict_ticker_with_model(ticker)
            if harga_now == 0 or harga_pred == 0:
                gagal.append(ticker)
                continue
            save_prediction_chart(ticker, df, future_preds)
            raw_return = (harga_pred_3bulan - harga_now) / harga_now
            expected_return = max(min(raw_return, 1.0), -0.5)
            results.append({
                'ticker': ticker,
                'current_price': harga_now,
                'predicted_price': harga_pred,
                'predicted_price_3mo': harga_pred_3bulan,
                'expected_return': expected_return,
            })

        for ticker in tickers_xiaomi:
            harga_now, harga_pred, harga_pred_3bulan, df, future_preds = predict_1810hk(ticker)
            if harga_now == 0 or harga_pred == 0:
                gagal.append(ticker)
                continue
            save_prediction_chart(ticker, df, future_preds)
            raw_return = (harga_pred_3bulan - harga_now) / harga_now
            expected_return = max(min(raw_return, 1.0), -0.5)
            results.append({
                'ticker': ticker,
                'current_price': harga_now,
                'predicted_price': harga_pred,
                'predicted_price_3mo': harga_pred_3bulan,
                'expected_return': expected_return,
            })

        for ticker in tickers_sony:
            harga_now, harga_pred, harga_pred_3bulan, df, future_preds = predict_sony(ticker)
            if harga_now == 0 or harga_pred == 0:
                gagal.append(ticker)
                continue
            save_prediction_chart(ticker, df, future_preds)
            raw_return = (harga_pred_3bulan - harga_now) / harga_now
            expected_return = max(min(raw_return, 1.0), -0.5)
            results.append({
                'ticker': ticker,
                'current_price': harga_now,
                'predicted_price': harga_pred,
                'predicted_price_3mo': harga_pred_3bulan,
                'expected_return': expected_return,
            })

        results = sorted(results, key=lambda x: x['expected_return'], reverse=True)
        results_positive = [r for r in results if r['expected_return'] > 0]

        # 🔧 Hitung alokasi hanya untuk saham dengan return positif
        df_alloc = optimize_portfolio(results_positive, dana)

        # 🧠 Gabungkan hasil alokasi kembali ke semua saham
        df_alloc_dict = df_alloc.set_index('ticker')[['allocation_percent', 'allocation_nominal', 'expected_profit']].to_dict('index')

        

        for r in results:
            if r['ticker'] in df_alloc_dict:
                r.update(df_alloc_dict[r['ticker']])
            else:
                r['allocation_percent'] = 0
                r['allocation_nominal'] = 0
                r['expected_profit'] = 0
                
        evaluasi = evaluate_portfolio(pd.DataFrame(results))
        session['last_results'] = results
        session['last_dana'] = dana

        if 'initial_results' not in session:
            session['initial_results'] = results.copy()

        pie_chart_path = generate_pie_chart(df_alloc)
        df_all = pd.DataFrame(results)
        df_view = format_df_for_display(df_all)

        # Cek jika ada ticker dipilih untuk tampilkan grafiknya
        chart_path = f"static/chart_13saham/prediksi_chart_{selected_ticker}.png" if selected_ticker else None
        if selected_ticker and not os.path.exists(chart_path):
            chart_path = None

        context = {
            'tables': [df_view.to_html(classes='table table-bordered', index=False, escape=False)],
            'pie_chart': pie_chart_path,
            'tickers': [r['ticker'] for r in results],
            'gagal_predict': gagal,
            'dana': dana,
            'selected_ticker': selected_ticker,
            'excluded_tickers': [],
            'evaluasi': evaluasi
        }
        if chart_path:
            context['chart_path'] = chart_path
            context['chart_paths'] = chart_path
            context['chart_ticker'] = selected_ticker

        return render_template('prediction.html',
            tables=[df_view.to_html(classes='table table-bordered', index=False, escape=False)],
            pie_chart=pie_chart_path,
            tickers=[r['ticker'] for r in results],
            gagal_predict=gagal,
            dana=dana,
            selected_ticker=selected_ticker,
            chart_paths=chart_path,
            chart_ticker=selected_ticker if chart_path else None,
            chart_path=chart_path,
            excluded_tickers=[],
            evaluasi=evaluasi
        )


    return render_template('prediction.html', evaluasi=None)


@app.route('/reallocation', methods=['POST'])
def reallocation():
    selected_ticker = request.form.get('selected_ticker')

    # ==== RESET ====
    if request.form.get('reset') == '1':
        results = session.get('initial_results', [])
        dana = session.get('last_dana', 0)
        excluded = []
        print("🔄 Reset aktif – semua saham & dana kembali ke kondisi awal.")

    # ==== REALLOKASI ====
    else:
        excluded = request.form.getlist('excluded_tickers')
        dana = float(request.form.get('dana_investasi'))
        results = session.get('last_results', [])
        print("✅ Excluded:", excluded)
        results = [r for r in results if r['ticker'] not in excluded]

    if not results:
        return render_template('prediction.html', error_message="Semua saham dikecualikan.")

    # Simpan hasil terbaru
    results = sorted(results, key=lambda x: x['expected_return'], reverse=True)
    session['last_results'] = results
    session['last_dana'] = dana

    # Hitung alokasi ulang
    df_alloc = optimize_portfolio(results, dana)

    # Gabungkan hasil alokasi ke dalam results
    df_alloc_dict = df_alloc.set_index('ticker')[['allocation_percent', 'allocation_nominal', 'expected_profit']].to_dict('index')
    for r in results:
        if r['ticker'] in df_alloc_dict:
            r.update(df_alloc_dict[r['ticker']])
        else:
            r['allocation_percent'] = 0
            r['allocation_nominal'] = 0
            r['expected_profit'] = 0

    # ⬅️ FORMAT BARU setelah update results
    df_view = format_df_for_display(pd.DataFrame(results))
    pie_chart_path = generate_pie_chart(pd.DataFrame(results))
    evaluasi = evaluate_portfolio(pd.DataFrame(results))

    return render_template('prediction.html',
        tables=[df_view.to_html(classes='table table-bordered', index=False, escape=False)],
        pie_chart=pie_chart_path,
        tickers=[r['ticker'] for r in results],
        gagal_predict=[],
        dana=dana,
        selected_ticker=selected_ticker,
        chart_ticker=None,
        excluded_tickers=excluded,
        evaluasi=evaluasi
    )



def predict_ticker_with_model(ticker, model_type='per_ticker'):
    model_path = f"models/model_{ticker}.pkl"
    scaler_path = f"models/scaler_{ticker}.pkl"
    features_path = f"models/features_{ticker}.pkl"

    if not all(os.path.exists(p) for p in [model_path, scaler_path, features_path]):
        print(f"[ERROR] Model files not found for {ticker}")
        return 0, 0, 0, pd.DataFrame(), []
    
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    features = joblib.load(features_path)

    df = yf.download(ticker, start=start_date, end=end_date, auto_adjust=False)
    sp500 = yf.download(sp500_ticker, start=start_date, end=end_date, auto_adjust=False)

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    if isinstance(sp500.columns, pd.MultiIndex):
        sp500.columns = sp500.columns.droplevel(1)

    if df.empty or len(df) < 100 or sp500.empty:
        print(f"[SKIP] {ticker}: Data tidak cukup atau SP500 kosong.")
        return 0, 0, 0, pd.DataFrame(), []


    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
    sp500 = sp500[['Close']].rename(columns={'Close': 'SP500_Close'})
    df = df.copy()
    df.index = df.index.tz_localize(None)
    sp500.index = sp500.index.tz_localize(None)
    df = df.join(sp500, how='inner')

    print(df.columns)
    print(df.head())
    
    df['Volume'] = np.log1p(df['Volume'])

    if ticker in tickers_13saham:
        df = generate_features_13saham(df)
    elif ticker in tickers_sony:
        df = generate_features_sony(df)
    else:
        print(f"[SKIP] {ticker}: Tidak dikenali dalam daftar saham.")
        return 0, 0, 0, pd.DataFrame(), []

    
    df.dropna(inplace=True)

    if df.empty:
        print(f"[SKIP] {ticker}: Data kosong setelah feature engineering.")
        return 0, 0, 0, pd.DataFrame(), []


    missing = set(features) - set(df.columns)
    if missing:
        print(f"[SKIP] {ticker}: Missing features -> {missing}")
        return 0, 0, 0, pd.DataFrame(), []


    X_latest = df.loc[:, features].iloc[[-1]]
    X_latest = X_latest[features]
    if X_latest.shape[1] != scaler.mean_.shape[0]:
        print(f"[SKIP] {ticker}: Shape mismatch. X_latest={X_latest.shape[1]}, scaler={scaler.mean_.shape[0]}")
        return 0, 0, 0, pd.DataFrame(), []


    try:
        X_scaled = scaler.transform(X_latest)
        pred = model.predict(X_scaled)[0]

        final_pred = pred
        last_close = df['Close'].iloc[-1].item()

        # === PREDIKSI 3 BULAN KE DEPAN ===
        future_predictions = []
        future_price = last_close
        future_date = df.index[-1]

        for _ in range(90):
            future_date += timedelta(days=1)
            while future_date.weekday() >= 5:
                future_date += timedelta(days=1)

            X_scaled_latest = scaler.transform(X_latest)
            pred = model.predict(X_scaled_latest)[0]
            ma5_latest = X_latest['MA5'].values[0]
            final_pred_ensemble = 0.7 * pred + 0.3 * ma5_latest

            future_price = final_pred_ensemble
            future_predictions.append(future_price)
            
            new_row = X_latest.copy()
            new_row['Lag4'] = new_row['Lag3']
            new_row['Lag3'] = new_row['Lag2']
            new_row['Lag2'] = new_row['Lag1']
            new_row['Lag1'] = future_price
            new_row['MA5'] = (new_row['MA5'] * 4 + future_price) / 5
            new_row['MA10'] = (new_row['MA10'] * 9 + future_price) / 10
            new_row['MA20'] = (new_row['MA20'] * 19 + future_price) / 20
            new_row['MA50'] = (new_row['MA50'] * 49 + future_price) / 50
            new_row['Momentum'] = future_price - new_row['Lag4']
            new_row['Lag1_ratio'] = new_row['Lag1'] / new_row['Lag2']
            new_row['Lag2_ratio'] = new_row['Lag3'] / new_row['Lag4']
            new_row['MA5_vs_EMA10'] = new_row['MA5'] - new_row['EMA10']
            new_row['MA5_to_MA20'] = new_row['MA5'] / new_row['MA20']
            new_row['EMA10_to_EMA50'] = new_row['EMA10'] / new_row['EMA50']
            new_row['WMA10_to_WMA20'] = new_row['WMA10'] / new_row['WMA20']
            new_row['3D_Trend'] = future_price - new_row['Lag3']
            new_row['Price_Change'] = future_price - new_row['Lag1']
            new_row['Return'] = new_row['Price_Change'] / new_row['Lag1']
            new_row['Day'] = future_date.day
            new_row['Month'] = future_date.month
            new_row['Year'] = future_date.year
            new_row['DayOfWeek'] = future_date.weekday()

            X_latest = new_row

        # Debug information
        print(f"\n===== DEBUG: {ticker} =====")
        print(f"Last Close: {last_close}")
        print(f"Raw Prediction: {pred}")
        print(f"Final Prediction: {final_pred}")

        return float(last_close), float(final_pred), float(future_price), df, future_predictions


    except Exception as e:
        print(f"[ERROR] {ticker}: {e}")
        return 0, 0, 0, pd.DataFrame(), []

def predict_ticker_13saham(ticker):
    return predict_ticker_with_model(ticker, model_type='per_ticker')

def predict_1810hk(ticker):
    df = yf.download(ticker, start=start_date, end=end_date)
    sp500 = yf.download(sp500_ticker, start=start_date, end=end_date)
    horizon_days = 90

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    if isinstance(sp500.columns, pd.MultiIndex):
        sp500.columns = sp500.columns.droplevel(1)

    if df.empty or len(df) < 100:
        return 0, 0

    df = df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
    sp500 = sp500[['Close']].rename(columns={'Close': 'SP500_Close'}).dropna()
    df = df.join(sp500, how='inner')
    df['Volume'] = np.log1p(df['Volume'])

    df = generate_features_sony(df)
    df['Target'] = df['Close'].shift(-1) - df['Close']
    df.dropna(inplace=True)

    features = [col for col in df.columns if col not in ['Target', 'Close']]
    X = df[features]
    y = df['Target']

    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    last_input = scaler.transform(X.iloc[[-1]])

    param_grid = {
        'n_estimators': [300],
        'learning_rate': [0.01],
        'max_depth': [4],
        'subsample': [0.8],
        'colsample_bytree': [0.9]
    }

    model = XGBRegressor(random_state=42)
    grid = GridSearchCV(model, param_grid, cv=TimeSeriesSplit(n_splits=3), scoring='neg_mean_squared_error')
    grid.fit(X_train_scaled, y_train)
    best_model = grid.best_estimator_

    predicted_delta = best_model.predict(last_input).item()
    last_close = df['Close'].iloc[-1]
    next_price = last_close + predicted_delta
    # === PREDIKSI 3 BULAN KE DEPAN ===
    future_predictions = []
    future_dates = []
    close_series = df['Close'].tolist() + [next_price]

    future_predictions.append(next_price)
    next_date = df.index[-1] + timedelta(days=1)
    while next_date.weekday() >= 5:
        next_date += timedelta(days=1)
    future_dates.append(next_date)

    last_known_close = last_close
    last_known_input = X.iloc[[-1]].copy()

    for i in range(1, horizon_days):
        scaled_input = scaler.transform(last_known_input)
        predicted_delta = best_model.predict(scaled_input)[0]
        predicted_price = last_known_close + predicted_delta

        future_predictions.append(predicted_price)
        next_date += timedelta(days=1)
        while next_date.weekday() >= 5:
            next_date += timedelta(days=1)
        future_dates.append(next_date)
        close_series.append(predicted_price)

        # Update fitur
        close_series_series = pd.Series(close_series)
        ema10 = EMAIndicator(close_series_series).ema_indicator().iloc[-1]
        rsi = RSIIndicator(close_series_series).rsi().iloc[-1]
        macd = MACD(close_series_series).macd().iloc[-1]
        bb = BollingerBands(close_series_series)
        bb_high = bb.bollinger_hband().iloc[-1]
        bb_low = bb.bollinger_lband().iloc[-1]

        new_row = last_known_input.copy()
        new_row['Lag1'] = predicted_price
        new_row['MA5'] = np.mean(close_series[-5:]) if len(close_series) >= 5 else np.nan
        new_row['MA10'] = np.mean(close_series[-10:]) if len(close_series) >= 10 else np.nan
        new_row['EMA10'] = EMAIndicator(pd.Series(close_series)).ema_indicator().iloc[-1]
        new_row['RSI'] = RSIIndicator(pd.Series(close_series)).rsi().iloc[-1]
        new_row['MACD'] = MACD(pd.Series(close_series)).macd().iloc[-1]
        new_row['BB_high'] = BollingerBands(pd.Series(close_series)).bollinger_hband().iloc[-1]
        new_row['BB_low'] = BollingerBands(pd.Series(close_series)).bollinger_lband().iloc[-1]
        new_row['Return'] = (predicted_price - last_known_close) / last_known_close
        new_row['RollingReturn5'] = pd.Series(close_series).pct_change().rolling(5).mean().iloc[-1]
        new_row['RollingReturn10'] = pd.Series(close_series).pct_change().rolling(10).mean().iloc[-1]
        new_row['Volatility5'] = pd.Series(close_series).pct_change().rolling(5).std().iloc[-1]
        new_row['Price_Level'] = pd.qcut(pd.Series(close_series), q=4, labels=False).iloc[-1]

        last_known_input = new_row
        last_known_close = predicted_price

    return float(last_close), float(next_price), float(future_predictions[-1]), df, future_predictions

def predict_sony(ticker):
    df = yf.download(ticker, start=start_date, end=end_date)
    sp500 = yf.download(sp500_ticker, start=start_date, end=end_date)

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    if isinstance(sp500.columns, pd.MultiIndex):
        sp500.columns = sp500.columns.droplevel(1)

    if df.empty or len(df) < 100:
        return 0, 0

    df = df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
    sp500 = sp500[['Close']].rename(columns={'Close': 'SP500_Close'}).dropna()
    df = df.join(sp500, how='inner')
    df['Volume'] = np.log1p(df['Volume'])

    df = generate_features_sony(df)
    df['Target'] = df['Close'].shift(-1) - df['Close']
    df.dropna(inplace=True)

    features = [col for col in df.columns if col not in ['Target', 'Close']]
    X = df[features]
    y = df['Target']

    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    last_input = scaler.transform(X.iloc[[-1]])

    param_grid = {
        'n_estimators': [300],
        'learning_rate': [0.01],
        'max_depth': [4],
        'subsample': [0.8],
        'colsample_bytree': [0.9]
    }

    model = XGBRegressor(random_state=42)
    grid = GridSearchCV(model, param_grid, cv=TimeSeriesSplit(n_splits=3), scoring='neg_mean_squared_error')
    grid.fit(X_train_scaled, y_train)
    best_model = grid.best_estimator_

    predicted_delta = best_model.predict(last_input).item()
    last_close = df['Close'].iloc[-1]
    next_price = last_close + predicted_delta

    # === Prediksi 3 bulan ke depan ===
    horizon_days = 90
    future_predictions = []
    future_dates = []
    close_series = df['Close'].tolist()

    last_known_close = last_close
    last_known_input = X.iloc[[-1]].copy()
    next_date = df.index[-1] + timedelta(days=1)
    while next_date.weekday() >= 5:
        next_date += timedelta(days=1)

    for i in range(horizon_days):
        scaled_input = scaler.transform(last_known_input)
        predicted_delta = best_model.predict(scaled_input)[0]

        # Hybrid ensemble
        ma5_latest = np.mean(close_series[-5:]) if len(close_series) >= 5 else last_known_close
        predicted_price = last_known_close + 0.6 * predicted_delta + 0.4 * (ma5_latest - last_known_close)

        # Clamp
        if predicted_price < 0.8 * last_close:
            predicted_price = 0.8 * last_close

        future_predictions.append(predicted_price)
        future_dates.append(next_date)
        close_series.append(predicted_price)

        next_date += timedelta(days=1)
        while next_date.weekday() >= 5:
            next_date += timedelta(days=1)

        # Update fitur
        new_row = last_known_input.copy()
        new_row['Lag1'] = predicted_price
        close_series_series = pd.Series(close_series)
        new_row['MA5'] = np.mean(close_series[-5:]) if len(close_series) >= 5 else np.nan
        new_row['MA10'] = np.mean(close_series[-10:]) if len(close_series) >= 10 else np.nan
        new_row['EMA10'] = EMAIndicator(close_series_series).ema_indicator().iloc[-1]
        new_row['RSI'] = RSIIndicator(close_series_series).rsi().iloc[-1]
        new_row['MACD'] = MACD(close_series_series).macd().iloc[-1]
        new_row['BB_high'] = BollingerBands(close_series_series).bollinger_hband().iloc[-1]
        new_row['BB_low'] = BollingerBands(close_series_series).bollinger_lband().iloc[-1]
        new_row['Return'] = (predicted_price - close_series[-2]) / close_series[-2] if len(close_series) > 1 else 0
        new_row['RollingReturn5'] = close_series_series.pct_change().rolling(5).mean().iloc[-1]
        new_row['RollingReturn10'] = close_series_series.pct_change().rolling(10).mean().iloc[-1]
        new_row['Volatility5'] = close_series_series.pct_change().rolling(5).std().iloc[-1]
        new_row['Price_Level'] = pd.qcut(close_series_series, q=4, labels=False).iloc[-1]

        last_known_input = new_row
        last_known_close = predicted_price

    return float(last_close), float(next_price), float(future_predictions[-1]), df, future_predictions

def generate_features_13saham(df):
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
    return df

def generate_features_sony(df):
    df['Lag1'] = df['Close'].shift(1)
    df['MA5'] = df['Close'].rolling(5).mean()
    df['MA10'] = df['Close'].rolling(10).mean()
    df['EMA10'] = EMAIndicator(close=df['Close'].squeeze(), window=10).ema_indicator()
    df['RSI'] = RSIIndicator(close=df['Close'].squeeze(), window=14).rsi()
    macd = MACD(close=df['Close'].squeeze())
    df['MACD'] = macd.macd()
    df['BB_high'] = BollingerBands(close=df['Close'].squeeze()).bollinger_hband()
    df['BB_low'] = BollingerBands(close=df['Close'].squeeze()).bollinger_lband()
    df['Return'] = df['Close'].pct_change()
    df['RollingReturn5'] = df['Return'].rolling(5).mean()
    df['RollingReturn10'] = df['Return'].rolling(10).mean()
    df['Volatility5'] = df['Close'].pct_change().rolling(5).std()
    df['Price_Level'] = pd.qcut(df['Close'].iloc[:, 0] if isinstance(df['Close'], pd.DataFrame) else df['Close'], q=4, labels=False)

    return df

def optimize_portfolio(results, dana):
    df = pd.DataFrame(results)
    df = df[df['expected_return'] > 0].copy()

    if df.empty:
        df['allocation_percent'] = 0
        df['allocation_nominal'] = 0
        df['expected_profit'] = 0
        return df

    tickers = df['ticker'].tolist()
    expected_returns = df['expected_return'].values

    # Ambil return historis
    returns_hist = []
    for ticker in tickers:
        data = yf.download(ticker, period="6mo")['Close'].pct_change().dropna()
        returns_hist.append(data)

    returns_df = pd.concat(returns_hist, axis=1)
    returns_df.columns = tickers
    returns_df.dropna(inplace=True)
    cov_matrix = returns_df.cov().values

    model = Model("Markowitz_PropReturn")
    model.setParam('OutputFlag', 0)

    # Variabel bobot portofolio
    weights = {
        t: model.addVar(lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS)
        for t in tickers
    }

    # Fungsi objektif: maksimalkan expected return
    model.setObjective(
        sum(weights[t] * r for t, r in zip(tickers, expected_returns)),
        GRB.MAXIMIZE
    )

    # Total alokasi = 100%
    model.addConstr(sum(weights.values()) == 1)

    # Batas risiko maksimum
    max_variance = 0.01
    portfolio_variance = sum(
        weights[tickers[i]] * weights[tickers[j]] * cov_matrix[i][j]
        for i in range(len(tickers))
        for j in range(len(tickers))
    )
    model.addConstr(portfolio_variance <= max_variance)

    # Tambah constraint agar alokasi proporsional terhadap return (±50%)
    total_return = df['expected_return'].sum()
    for t, r in zip(tickers, df['expected_return']):
        prop_weight = r / total_return
        model.addConstr(weights[t] >= prop_weight * 0.5)
        model.addConstr(weights[t] <= prop_weight * 1.5)

    # Solve
    model.optimize()

    if model.status == GRB.OPTIMAL:
        df['allocation_percent'] = [weights[t].X for t in tickers]
    else:
        print("[WARNING] Gurobi gagal optimasi.")
        df['allocation_percent'] = 0

    df['allocation_nominal'] = df['allocation_percent'] * dana
    df['expected_profit'] = df['allocation_nominal'] * df['expected_return']
    return df


def generate_pie_chart(df):
    # Hapus saham dengan alokasi 0
    df_filtered = df[df['allocation_percent'] > 0].copy()

    if df_filtered.empty:
        return None  # Gak usah gambar chart kalau kosong

    plt.figure(figsize=(8, 8))
    plt.pie(df_filtered['allocation_percent'], labels=df_filtered['ticker'], autopct='%1.1f%%')
    plt.title('Distribusi Portofolio Saham')
    plt.tight_layout()
    pie_chart_path = os.path.join('static/img', 'portfolio_pie.png')
    plt.savefig(pie_chart_path)
    plt.close()
    return pie_chart_path


def save_prediction_chart(ticker, df, future_predictions):
    os.makedirs('static/chart_13saham', exist_ok=True)
    output_path = os.path.join('static', 'chart_13saham', f'prediksi_chart_{ticker}.png')

    # Ambil 30 hari terakhir data aktual
    df_plot = df[-30:].copy()
    df_plot_dates = df_plot.index.strftime('%Y-%m-%d').tolist()

    # Buat daftar tanggal prediksi (skip weekend)
    future_dates = []
    future_date = df.index[-1]
    for _ in range(len(future_predictions)):
        future_date += timedelta(days=1)
        while future_date.weekday() >= 5:  # Sabtu/Minggu
            future_date += timedelta(days=1)
        future_dates.append(future_date.strftime('%Y-%m-%d'))

    extended_values = df_plot['Close'].tolist() + [float(p) for p in future_predictions]
    split_index = len(df_plot_dates) - 1

    # Ambil bulan untuk penjelasan
    actual_start = pd.to_datetime(df_plot_dates[0]).strftime('%b %Y')
    actual_end = pd.to_datetime(df_plot_dates[-1]).strftime('%b %Y')
    pred_start = pd.to_datetime(future_dates[0]).strftime('%b %Y')
    pred_end = pd.to_datetime(future_dates[-1]).strftime('%b %Y')

    label_keterangan = f'Aktual: {actual_start}–{actual_end}, Prediksi: {pred_start}–{pred_end}'

    # Plot tanpa xticks
    plt.figure(figsize=(12, 5))
    plt.plot(range(len(extended_values)), extended_values, label='Harga', marker='o', color='blue')
    plt.axvline(x=split_index, color='red', linestyle='--', label='Mulai Prediksi')
    plt.fill_between(range(split_index+1), min(extended_values), max(extended_values), color='gray', alpha=0.1, label='Aktual')
    plt.fill_between(range(split_index, len(extended_values)), min(extended_values), max(extended_values), color='orange', alpha=0.1, label='Prediksi')
    
    plt.title(f'Harga Penutupan {ticker}: Aktual vs Prediksi 3 Bulan')
    plt.xlabel(label_keterangan)
    plt.ylabel('Harga Penutupan')
    plt.xticks([])  # Hapus label tanggal
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    return output_path

def format_df_for_display(df):
    df['allocation_nominal'] = df['allocation_nominal'].apply(lambda x: f"IDR {x:,.0f}")
    df['current_price'] = df['current_price'].apply(lambda x: f"{x:.2f}")
    df['predicted_price'] = df['predicted_price'].apply(lambda x: f"{x:.2f}")
    df['predicted_price_3mo'] = df['predicted_price_3mo'].apply(lambda x: f"{x:.2f}")
    df['expected_return'] = df['expected_return'].apply(lambda x: f"{x*100:.2f}%")
    df['expected_profit'] = df['expected_profit'].apply(lambda x: f"IDR {x:,.0f}")
    


    df = df.rename(columns={
        'ticker': 'Saham',
        'current_price': 'Harga Saat Ini',
        'predicted_price': 'Prediksi Besok',
        'predicted_price_3mo': 'Prediksi 3 Bulan',
        'expected_return': 'Return (%)',
        'allocation_nominal': 'Alokasi (IDR)',
        'expected_profit': 'Estimasi Profit (IDR)',
        
    })

    return df[['Saham', 'Harga Saat Ini', 'Prediksi Besok', 'Prediksi 3 Bulan',
               'Return (%)', 'Alokasi (IDR)', 'Estimasi Profit (IDR)']]

@app.route('/evaluasi', methods=['GET'])
def evaluasi_portofolio():
    if 'last_results' not in session or 'last_dana' not in session:
        return "Tidak ada hasil prediksi dan alokasi sebelumnya untuk dievaluasi."

    results = session['last_results']
    dana = session['last_dana']

    tickers = [r['ticker'] for r in results if r['allocation_percent'] > 0]
    start_eval = (datetime.today() - timedelta(days=90)).strftime('%Y-%m-%d')
    end_eval = datetime.today().strftime('%Y-%m-%d')

    realized_returns = {}
    for t in tickers:
        try:
            data = yf.download(t, start=start_eval, end=end_eval)
            if len(data) < 2:
                continue
            p_awal = data['Close'].iloc[0]
            p_akhir = data['Close'].iloc[-1]
            realized_returns[t] = (p_akhir - p_awal) / p_awal
        except:
            realized_returns[t] = 0.0

    total_realized_return = 0
    total_expected_return = 0
    for r in results:
        if r['ticker'] in realized_returns:
            w = r['allocation_percent']
            total_realized_return += w * realized_returns[r['ticker']]
            total_expected_return += w * r['expected_return']

    mape = np.mean([
        abs((r['expected_return'] - realized_returns.get(r['ticker'], 0)) / (realized_returns.get(r['ticker'], 1e-6)))
        for r in results if r['allocation_percent'] > 0
    ])

    output = f"<h3>Evaluasi Portofolio</h3>"
    output += f"<p>Total Realized Return: {total_realized_return:.4f} ({total_realized_return*100:.2f}%)</p>"
    output += f"<p>Total Expected Return: {total_expected_return:.4f} ({total_expected_return*100:.2f}%)</p>"
    output += f"<p>MAPE: {mape:.4f} ({mape*100:.2f}%)</p>"
    return output

if __name__ == '__main__':
    app.run(debug=True)