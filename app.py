from flask import Flask, render_template, request, session, send_file, jsonify
from flask_session import Session
import joblib
import pandas as pd
import numpy as np
import yfinance as yf
import hashlib
import json
import redis
from gurobipy import Model, GRB, quicksum
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

# === KONFIGURASI SERVER-SIDE SESSION ===
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SESSION_FILE_DIR'] = './session_data'
app.config['SESSION_PERMANENT'] = False
app.config['SESSION_USE_SIGNER'] = True
app.config['SESSION_KEY_PREFIX'] = 'porto:'
app.config['SESSION_FILE_THRESHOLD'] = 10000
Session(app)

# === Daftar Ticker ===
tickers_13saham = ['NVDA', 'AAPL', 'GOOGL', '005930.KS', '000660.KQ',
                   'AMD', 'TSLA', 'ORCL', 'AMZN', 'INTC', 'MSFT', 'GOTO.JK', '9988.HK', '0700.HK']
tickers_xiaomi = ['1810.HK']
tickers_sony = ['SONY']
sp500_ticker = '^GSPC'
start_date = '2023-01-01'
end_date = '2025-01-01'

# === Kelas MarkowitzPortfolio (Ditempatkan di File Utama) ===
class MarkowitzPortfolio:
    def __init__(self, results, dana, max_risk=0.01):
        self.dana = dana
        self.max_risk = max_risk
        self.results = [r for r in results if isinstance(r.get('future_predictions'), list) and len(r['future_predictions']) > 1]
        self.tickers = [r['ticker'] for r in self.results]
        self.P = min(len(r['future_predictions']) for r in self.results) - 1
        if self.P < 1:
            raise ValueError("Jumlah prediksi terlalu sedikit (P < 1). Minimal butuh 2 data prediksi per saham.")
        self.mu_i = self._calculate_expected_returns()
        self.tickers = [r['ticker'] for r in self.results]
        self.results = [r for r in self.results if r['ticker'] in self.tickers]
        self.S_matrix = self._calculate_covariance_matrix()
    
    def _calculate_expected_returns(self):
        """Menggunakan expected_return yang sudah dihitung di results"""
        mu_dict = {r['ticker']: r.get('expected_return', 0) or 0 for r in self.results}
        return pd.Series(mu_dict)
    
    def _calculate_covariance_matrix(self):
        """Menghitung matriks covariance S"""
        returns_data = {}
        for r in self.results:
            prices = np.array(r['future_predictions'][:self.P+1])
            returns = (prices[1:] - prices[:-1]) / prices[:-1]
            returns_data[r['ticker']] = returns

        n_tickers = len(self.tickers)
        S_matrix = np.zeros((n_tickers, n_tickers))

        for i, ticker_k in enumerate(self.tickers):
            for j, ticker_l in enumerate(self.tickers):
                r_k = returns_data[ticker_k]
                r_l = returns_data[ticker_l]
                mu_k = np.mean(r_k)
                mu_l = np.mean(r_l)
                covariance = np.sum((r_k - mu_k) * (r_l - mu_l)) / self.P
                S_matrix[i, j] = covariance

        print(f"\n return: {self.mu_i}")
        print(f"\n covarians matriks: {S_matrix}")                
        return pd.DataFrame(S_matrix, index=self.tickers, columns=self.tickers)
    
    def _get_max_alloc_per_ticker(self):
        n = len(self.tickers)
        if n <= 2:
            return 1.0
        elif n <= 3:
            return 0.7
        else:
            return 0.5
    
    def optimize(self):
        # MIQCP-Markowitz optimization sesuai persamaan (4) dalam dokumen:
        # maximize: [xi]ᵀ[μi]
        # subject to: [xi]ᵀS[xi] ≤ σmax
        #            xi ≥ 0
        #            Σxi = 1
        model = Model("MIQCP_Markowitz")
        model.setParam("OutputFlag", 0)
        
        # Variabel keputusan: xi (persentase dana untuk setiap saham)
        x = {t: model.addVar(lb=0.0, ub=1.0, name=f"x_{t}") for t in self.tickers}
        model.update()
        # Batasi dominasi saham
        max_alloc = self._get_max_alloc_per_ticker()
        top_return = self.mu_i.max()
        for t in self.tickers:
            # # Batasi alokasi maksimum per ticker
            # model.addConstr(x[t] <= max_alloc, name=f"max_alloc_{t}")
            
            # # Tetap kasih alokasi minimum kalau return tinggi
            # if self.mu_i[t] >= 0.9 * top_return:
            #     model.addConstr(x[t] <= max_alloc, name=f"max_alloc_top_{t}")
            #     model.addConstr(x[t] >= 0.05, name=f"min_alloc_top_{t}")
            # else:
            #     model.addConstr(x[t] <= 0.2, name=f"max_alloc_low_{t}")
            #     model.addConstr(x[t] >= 0.01, name=f"min_alloc_low_{t}")
            model.addConstr(x[t] >= 0, name=f"Minumum")
        
        # Fungsi Objektif (4a): maximize [xi]ᵀ[μi]
        model.setObjective(
            quicksum(self.mu_i[t] * x[t] for t in self.tickers), 
            GRB.MAXIMIZE
        )      

        # Constraint (4d): Σxi = 1 (total alokasi = 100%)
        model.addConstr(quicksum(x[t] for t in self.tickers) == 1)  

        # Constraint (4b): [xi]ᵀS[xi] ≤ σmax (batasan risiko)
        portfolio_variance = quicksum(
            x[i] * x[j] * self.S_matrix.loc[i, j]
            for i in self.tickers for j in self.tickers
        )
        model.addConstr(portfolio_variance <= self.max_risk)      

        print(f"\n[DEBUG] Jumlah ticker masuk ke optimasi: {len(self.tickers)}")
        print(f"[DEBUG] Daftar ticker: {self.tickers}")
        print(f"[DEBUG] Expected Return semua saham:")
        print(self.mu_i)
        # Constraint (4c): xi ≥ 0 (sudah diatur di addVar dengan lb=0.0)       
        # Solve optimization
        model.optimize()
        if model.Status != GRB.OPTIMAL:
            print("[ERROR] Model tidak optimal")
            return {"weights": {}, "expected_return": 0, "risk": 0}
        
        # Extract optimal weights
        weights = {t: x[t].X for t in self.tickers if x[t].X > 1e-4}

        # Hitung expected return portfolio: [xi]ᵀ[μi]
        expected_return = sum(self.mu_i[t] * weights[t] for t in weights)
        print(f"[DEBUG] Expected Return : {expected_return:.6f}")
        
        # Hitung portfolio risk: [xi]ᵀS[xi]
        portfolio_variance = sum(
            weights[i] * weights[j] * self.S_matrix.loc[i, j] 
            for i in weights for j in weights
        )
        print(f"[DEBUG] Variansi Portofolio (σ²) : {portfolio_variance:.6f}")

        risk = np.sqrt(portfolio_variance)
        print(f"[DEBUG] Standar Deviasi Portofolio (σ) : {risk:.6f}")

        # Hitung Sharpe Ratio
        R_f = 0.03 / 252
        sharpe_ratio = (expected_return - R_f) / risk if risk > 0 else 0
        print(f"[DEBUG] Sharpe Ratio : {sharpe_ratio:.6f}")

        # Cetak ke terminal
        print(f"\n📊 Expected Return : {expected_return:.6f}")
        print(f"📉 Portfolio Risk  : {risk:.6f}")
        print(f"📈 Sharpe Ratio    : {sharpe_ratio:.4f}")

        # === Simpan hasil alokasi portofolio ===
        df_output = pd.DataFrame([
            {
                "Ticker": t,
                "Weight (%)": round(weights[t] * 100, 2),
                "Alokasi Dana": round(weights[t] * self.dana, 2),
                "Expected Return": round(self.mu_i[t], 6)
            }
            for t in weights
        ])
        df_output.loc["Total"] = [
            "Total",
            round(df_output["Weight (%)"].sum(), 2),
            round(df_output["Alokasi Dana"].sum(), 2),
            round(expected_return, 6)
        ]
        df_output.to_csv("hasil_optimasi_portofolio.csv", index=False)

        # === Simpan ringkasan metrik portofolio ===
        df_summary = pd.DataFrame({
            "Metric": ["Expected Return", "Risk (Std Dev)", "Sharpe Ratio"],
            "Value": [expected_return, risk, sharpe_ratio]
        })
        df_summary.to_csv("ringkasan_portofolio.csv", index=False)

        # === Simpan covariance matrix (opsional) ===
        self.S_matrix.to_csv("covariance_matrix.csv")

        return {
            "weights": weights,
            "expected_return": expected_return,
            "risk": risk,
            "sharpe_ratio": sharpe_ratio,
            "allocations": {t: weights[t] * self.dana for t in weights}
        }
def load_model_by_ticker(ticker):
    model_path = f"models/model_{ticker}.pkl"
    scaler_path = f"models/scaler_{ticker}.pkl"
    features_path = f"models/features_{ticker}.pkl"

    if not all(os.path.exists(p) for p in [model_path, scaler_path, features_path]):
        print(f"[ERROR] Model files not found for {ticker}")
        return None, None, None

    try:
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        features = joblib.load(features_path)
    except Exception as e:
        print(f"[ERROR] Gagal memuat model {ticker}: {e}")
        return None, None, None

    return model, scaler, features

@app.route('/')
def home():
    return render_template('home.html')
class DataCache:
    def __init__(self, cache_dir='./cache_data'):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
    
    def _generate_key(self, data):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        data_str = json.dumps(data, sort_keys=True, default=str)
        hash_key = hashlib.md5(data_str.encode()).hexdigest()[:8]
        return f"{timestamp}_{hash_key}"
    
    def save(self, data, key=None):
        if key is None:
            key = self._generate_key(data)
        cache_file = os.path.join(self.cache_dir, f"{key}.json")
        with open(cache_file, 'w') as f:
            json.dump(data, f, default=str)
        return key
    
    def load(self, key):
        cache_file = os.path.join(self.cache_dir, f"{key}.json")
        if os.path.exists(cache_file):
            with open(cache_file, 'r') as f:
                return json.load(f)
        return None
    
    def cleanup_old_files(self, days=7):
        cutoff = datetime.now() - timedelta(days=days)
        for filename in os.listdir(self.cache_dir):
            if filename.endswith('.json'):
                file_path = os.path.join(self.cache_dir, filename)
                if os.path.getctime(file_path) < cutoff.timestamp():
                    os.remove(file_path)

cache = DataCache()

def evaluasi_portfolio(df):
    weights = {r['ticker']: r['allocation_percent'] for _, r in df.iterrows() if r['allocation_percent'] > 0}
    mu_i = df.set_index('ticker')['expected_return']
    weight_array = np.array([weights[t] for t in mu_i.index if t in weights])
    mu_array = mu_i[mu_i.index.isin(weights.keys())].values
    expected_return = float(np.dot(weight_array, mu_array)) if len(weight_array) > 0 else 0.0

    future_returns = {
        r['ticker']: r['future_predictions'] for _, r in df.iterrows()
        if r['allocation_percent'] > 0 and isinstance(r['future_predictions'], list) and len(r['future_predictions']) > 1
    }
    if future_returns:
        log_return_df = pd.DataFrame({
            t: np.log(np.array(future_returns[t])[1:] / np.array(future_returns[t])[:-1])
            for t in future_returns
        })
        cov_matrix = log_return_df.cov()
        risk = float(np.sqrt(np.dot(weight_array.T, np.dot(cov_matrix.values, weight_array))))
    else:
        risk = 0.0

    R_f = 0.03 / 252
    sharpe_ratio = (expected_return - R_f) / risk if risk > 0 else 0

    return {
        'expected_return': expected_return,
        'risk': risk,
        'sharpe_ratio': sharpe_ratio,
        'num_assets': len(weights)
    }

@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    if request.method == 'POST':
        dana_input_raw = request.form.get('dana_investasi')
        dana_input = float(dana_input_raw) if dana_input_raw else 0
        selected_ticker = request.form.get('selected_ticker')

        if selected_ticker and 'results_cache_key' in session and 'last_dana' in session:
            cached_results = cache.load(session['results_cache_key'])
            if cached_results:
                df_alloc = optimize_portfolio(cached_results, session['last_dana'])
                pie_chart_path = generate_pie_chart(df_alloc)
                df_view = format_df_for_display(df_alloc)
                evaluasi = evaluasi_portfolio(pd.DataFrame(cached_results))

            chart_path = f"static/chart_13saham/prediksi_chart_{selected_ticker}.png"
            if not os.path.exists(chart_path):
                chart_path = None
            df_all = pd.DataFrame(cached_results)
            df_all['expected_profit'] = pd.to_numeric(df_all.get('expected_profit', 0), errors='coerce').fillna(0)
            total_profit = df_all['expected_profit'].sum()

            context = {
                'tables': [df_view.to_html(classes='table table-bordered', index=False, escape=False)],
                'pie_chart': pie_chart_path,
                'tickers': [r['ticker'] for r in cached_results] if cached_results else [],
                'gagal_predict': [],
                'dana': session['last_dana'],
                'selected_ticker': selected_ticker,
                'excluded_tickers': [],
                'chart_path': chart_path,
                'chart_paths': chart_path,
                'chart_ticker': selected_ticker if chart_path else None,
                'evaluasi': evaluasi,
                'total_profit': total_profit
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
            avg_daily_return = calculate_expected_return_from_preds(future_preds)
            expected_return = max(avg_daily_return, 0.0)
            results.append({
                'ticker': ticker,
                'current_price': harga_now,
                'predicted_price': harga_pred,
                'predicted_price_3mo': harga_pred_3bulan,
                'expected_return': expected_return,
                'future_predictions': future_preds,
            })

        for ticker in tickers_xiaomi:
            harga_now, harga_pred, harga_pred_3bulan, df, future_preds = predict_1810hk(ticker)
            if harga_now == 0 or harga_pred == 0:
                gagal.append(ticker)
                continue
            save_prediction_chart(ticker, df, future_preds)
            avg_daily_return = calculate_expected_return_from_preds(future_preds)
            expected_return = max(avg_daily_return, 0.0)
            results.append({
                'ticker': ticker,
                'current_price': harga_now,
                'predicted_price': harga_pred,
                'predicted_price_3mo': harga_pred_3bulan,
                'expected_return': expected_return,
                'future_predictions': future_preds,
            })

        for ticker in tickers_sony:
            harga_now, harga_pred, harga_pred_3bulan, df, future_preds = predict_sony(ticker)
            if harga_now == 0 or harga_pred == 0:
                gagal.append(ticker)
                continue
            save_prediction_chart(ticker, df, future_preds)
            avg_daily_return = calculate_expected_return_from_preds(future_preds)
            expected_return = max(avg_daily_return, 0.0)
            results.append({
                'ticker': ticker,
                'current_price': harga_now,
                'predicted_price': harga_pred,
                'predicted_price_3mo': harga_pred_3bulan,
                'expected_return': expected_return,
                'future_predictions': future_preds,
            })

        results = sorted(results, key=lambda x: x['expected_return'], reverse=True)
        # results_positive = [r for r in results if r['expected_return'] > 0]
        results_positive = results
        df_alloc = optimize_portfolio(results_positive, dana)
        df_alloc_dict = df_alloc.set_index('ticker')[['allocation_percent', 'allocation_nominal', 'expected_profit']].to_dict('index')

        for r in results:
            if r['ticker'] in df_alloc_dict:
                r.update(df_alloc_dict[r['ticker']])
            else:
                r['allocation_percent'] = 0
                r['allocation_nominal'] = 0
                r['expected_profit'] = 0
                
        evaluasi = evaluasi_portfolio(pd.DataFrame(results))
        cache_key = cache.save(results)
        session['results_cache_key'] = cache_key
        session['last_dana'] = dana
        session['last_tickers'] = [r['ticker'] for r in results]
        session['last_evaluasi'] = {
            'expected_return': float(evaluasi['expected_return']),
            'risk': float(evaluasi['risk']),
            'sharpe_ratio': float(evaluasi['sharpe_ratio']),
            'num_assets': evaluasi['num_assets']
        }

        if 'initial_cache_key' not in session:
            initial_cache_key = cache.save(results)
            session['initial_cache_key'] = initial_cache_key

        pie_chart_path = generate_pie_chart(df_alloc)
        df_all = pd.DataFrame(results)
        df_view = format_df_for_display(df_all.copy())

        df_all['expected_profit'] = pd.to_numeric(df_all['expected_profit'], errors='coerce').fillna(0)
        total_profit = df_all['expected_profit'].sum()
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
            'evaluasi': evaluasi,
            'total_profit': total_profit
        }
        if chart_path:
            context['chart_path'] = chart_path
            context['chart_paths'] = chart_path
            context['chart_ticker'] = selected_ticker
        
        base_date = pd.Timestamp(end_date) + timedelta(days=1)
        while base_date.weekday() >= 5:
            base_date += timedelta(days=1)

        future_dates = []
        curr_date = base_date
        for _ in range(63):
            while curr_date.weekday() >= 5:
                curr_date += timedelta(days=1)
            future_dates.append(curr_date.strftime('%Y-%m-%d'))
            curr_date += timedelta(days=1)

        pred_dict = {}
        for r in results:
            ticker = r['ticker']
            preds = r['future_predictions']
            pred_dict[ticker] = preds[:63] if len(preds) >= 63 else preds + [None] * (63 - len(preds))

        df_preds = pd.DataFrame(pred_dict, index=future_dates)
        df_preds.index.name = 'Date'
        pred_path = f"data_prediksi/prediksi_harian.csv"
        os.makedirs("data_prediksi", exist_ok=True)
        df_preds.to_csv(pred_path)
        print(f"✅ Prediksi harian 3 bulan ke depan disimpan: {pred_path}")

        print(f"[DEBUG FINAL] Expected Return: {evaluasi['expected_return']}")
        print(f"[DEBUG FINAL] Risk (Std Dev): {evaluasi['risk']}")
        print(f"[DEBUG FINAL] Sharpe Ratio: {evaluasi['sharpe_ratio']}")

        return render_template('prediction.html', **context)

    return render_template('prediction.html', evaluasi=None, total_profit=0)

@app.route('/reallocation', methods=['POST'])
def reallocation():
    try:
        selected_ticker = request.form.get('selected_ticker')
        results = []
        excluded = []

        if request.form.get('reset') == '1':
            if 'initial_cache_key' in session:
                results = cache.load(session['initial_cache_key']) or []
            dana = session.get('last_dana', 0)
            print("🔄 Reset aktif – semua saham & dana kembali ke kondisi awal.")
            session['results_cache_key'] = cache.save(results)

            df_alloc = optimize_portfolio(results, dana)
            df_alloc_dict = df_alloc.set_index('ticker')[['allocation_percent', 'allocation_nominal', 'expected_profit']].to_dict('index')

            for r in results:
                if r['ticker'] in df_alloc_dict:
                    r.update(df_alloc_dict[r['ticker']])
                else:
                    r['allocation_percent'] = 0
                    r['allocation_nominal'] = 0
                    r['expected_profit'] = 0

            df_view = format_df_for_display(pd.DataFrame(results))
            pie_chart_path = generate_pie_chart(pd.DataFrame(results))
            evaluasi = evaluasi_portfolio(pd.DataFrame(results))

            df_all = pd.DataFrame(results)
            df_all['expected_profit'] = pd.to_numeric(df_all.get('expected_profit', 0), errors='coerce').fillna(0)
            total_profit = df_all['expected_profit'].sum()

            return render_template('prediction.html',
                tables=[df_view.to_html(classes='table table-bordered', index=False, escape=False)],
                pie_chart=pie_chart_path,
                tickers=[r['ticker'] for r in results],
                gagal_predict=[],
                dana=dana,
                selected_ticker=selected_ticker,
                chart_ticker=None,
                excluded_tickers=[],
                evaluasi=evaluasi,
                total_profit=total_profit
            )

        excluded = request.form.getlist('excluded_tickers')
        dana = float(request.form.get('dana_investasi', 0))

        if 'results_cache_key' in session:
            results_raw = cache.load(session['results_cache_key']) or []
        else:
            results_raw = []

        print(f"Total saham dari cache: {len(results_raw)}")
        print(f"Excluded tickers: {excluded}")

        results = []
        for r in results_raw:
            if r['ticker'] in excluded:
                print(f"[SKIP] {r['ticker']}: Dikecualikan oleh user")
                continue
            if (r['expected_return'] <= 0 or 
                'future_predictions' not in r or 
                not isinstance(r['future_predictions'], list) or 
                len(r['future_predictions']) < 3):
                continue
            results.append(r)

        if not results:
            error_msg = "❌ Tidak ada saham yang tersedia untuk optimasi."
            return render_template('prediction.html',
                error_message=error_msg,
                total_profit=0,
                evaluasi=None,
                tables=[],
                pie_chart=None,
                tickers=[],
                gagal_predict=[],
                dana=dana,
                selected_ticker=selected_ticker,
                chart_ticker=None,
                excluded_tickers=[]
            )

        results = sorted(results, key=lambda x: x['expected_return'], reverse=True)
        new_cache_key = cache.save(results)
        session['results_cache_key'] = new_cache_key
        session['last_dana'] = dana

        df_alloc = optimize_portfolio(results, dana)
        if df_alloc.empty:
            error_msg = "❌ Optimasi portfolio gagal."
            return render_template('prediction.html', error_message=error_msg)

        df_alloc_dict = df_alloc.set_index('ticker')[['allocation_percent', 'allocation_nominal', 'expected_profit']].to_dict('index')
        for r in results:
            if r['ticker'] in df_alloc_dict:
                r.update(df_alloc_dict[r['ticker']])
            else:
                r['allocation_percent'] = 0
                r['allocation_nominal'] = 0
                r['expected_profit'] = 0

        df_view = format_df_for_display(pd.DataFrame(results))
        pie_chart_path = generate_pie_chart(pd.DataFrame(results))
        evaluasi = evaluasi_portfolio(pd.DataFrame(results))

        df_all = pd.DataFrame(results)
        df_all['expected_profit'] = pd.to_numeric(df_all.get('expected_profit', 0), errors='coerce').fillna(0)
        total_profit = df_all['expected_profit'].sum()


        return render_template('prediction.html',
            tables=[df_view.to_html(classes='table table-bordered', index=False, escape=False)],
            pie_chart=pie_chart_path,
            tickers=[r['ticker'] for r in results],
            gagal_predict=[],
            dana=dana,
            selected_ticker=selected_ticker,
            chart_ticker=None,
            excluded_tickers=excluded,
            evaluasi=evaluasi,
            total_profit=total_profit
        )

    except Exception as e:
        error_msg = f"❌ Error pada reallocation: {str(e)}"
        return render_template('prediction.html', error_message=error_msg)

def predict_ticker_with_model(ticker, model_type='per_ticker'):
    model, scaler, features = load_model_by_ticker(ticker)
    if not all([model, scaler, features]):
        print(f"[SKIP] {ticker}: Model/Scaler/Features not loaded.")
        return 0, 0, 0, pd.DataFrame(), []

    df = yf.download(ticker, start=start_date, end=end_date, auto_adjust=False)
    print(f"[DEBUG] Last date of {ticker}: {df.index[-1]}")
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

        future_predictions = []
        future_price = last_close
        future_date = df.index[-1]

        for _ in range(63):
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

        print(f"\n===== DEBUG: {ticker} =====")
        print(f"Last Close: {last_close}")
        print(f"Raw Prediction: {pred}")
        print(f"Final Prediction: {final_pred}")

        return float(last_close), float(final_pred), float(future_price), df, future_predictions

    except Exception as e:
        print(f"[ERROR] {ticker}: {e}")
        return 0, 0, 0, pd.DataFrame(), []

def predict_1810hk(ticker):
    df = yf.download(ticker, start=start_date, end=end_date)
    print(f"[DEBUG] Last date of {ticker}: {df.index[-1]}")
    sp500 = yf.download(sp500_ticker, start=start_date, end=end_date)
    horizon_days = 63

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    if isinstance(sp500.columns, pd.MultiIndex):
        sp500.columns = sp500.columns.droplevel(1)

    if df.empty or len(df) < 100:
        return 0, 0, 0, pd.DataFrame(), []

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

        close_series_series = pd.Series(close_series)
        new_row = last_known_input.copy()
        new_row['Lag1'] = predicted_price
        new_row['MA5'] = np.mean(close_series[-5:]) if len(close_series) >= 5 else np.nan
        new_row['MA10'] = np.mean(close_series[-10:]) if len(close_series) >= 10 else np.nan
        new_row['EMA10'] = EMAIndicator(close_series_series).ema_indicator().iloc[-1]
        new_row['RSI'] = RSIIndicator(close_series_series).rsi().iloc[-1]
        new_row['MACD'] = MACD(close_series_series).macd().iloc[-1]
        new_row['BB_high'] = BollingerBands(close_series_series).bollinger_hband().iloc[-1]
        new_row['BB_low'] = BollingerBands(close_series_series).bollinger_lband().iloc[-1]
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
    print(f"[DEBUG] Last date of {ticker}: {df.index[-1]}")
    sp500 = yf.download(sp500_ticker, start=start_date, end=end_date)

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    if isinstance(sp500.columns, pd.MultiIndex):
        sp500.columns = sp500.columns.droplevel(1)

    if df.empty or len(df) < 100:
        return 0, 0, 0, pd.DataFrame(), []

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

    horizon_days = 63
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
        ma5_latest = np.mean(close_series[-5:]) if len(close_series) >= 5 else last_known_close
        predicted_price = last_known_close + 0.6 * predicted_delta + 0.4 * (ma5_latest - last_known_close)

        if predicted_price < 0.8 * last_close:
            predicted_price = 0.8 * last_close

        future_predictions.append(predicted_price)
        future_dates.append(next_date)
        close_series.append(predicted_price)

        next_date += timedelta(days=1)
        while next_date.weekday() >= 5:
            next_date += timedelta(days=1)

        close_series_series = pd.Series(close_series)
        new_row = last_known_input.copy()
        new_row['Lag1'] = predicted_price
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

def calculate_expected_return_from_preds(future_predictions):
    if not future_predictions or len(future_predictions) < 2:
        return 0
    prices = np.array(future_predictions)
    returns = (prices[1:] - prices[:-1]) / prices[:-1]
    return np.mean(returns)

def optimize_portfolio(results, dana):
    df = pd.DataFrame(results)
    df['expected_return'] = pd.to_numeric(df['expected_return'], errors='coerce').fillna(0)
    (df['future_predictions'].apply(lambda x: isinstance(x, list) and len(x) > 1))

    if df.empty:
        print("[ERROR] Tidak ada saham valid untuk optimasi.")
        df['allocation_percent'] = 0
        df['allocation_nominal'] = 0
        df['expected_profit'] = 0
        return df

    try:
        optimizer = MarkowitzPortfolio(results=df.to_dict('records'), dana=dana, max_risk=0.03)
        result = optimizer.optimize()
        weights = result.get('weights', {})
        if not weights:
            raise Exception("Optimasi gagal: tidak ada bobot ditemukan")
        print(f"✅ Optimasi sukses: Return = {result['expected_return']:.4f}, Risk = {result['risk']:.4f}")
    except Exception as e:
        print(f"[ERROR] Gagal optimasi MIQCP: {e}")
        n = len(df)
        weights = {r['ticker']: 1.0 / n for r in df.to_dict('records')}

    df = df[df['ticker'].isin(weights.keys())].copy()
    df['allocation_percent'] = df['ticker'].apply(lambda t: weights.get(t, 0))
    df['allocation_nominal'] = df['allocation_percent'] * dana
    df['expected_profit'] = df['allocation_nominal'] * df['expected_return']

    print("=== Final Allocation ===")
    print(df[['ticker', 'allocation_percent', 'allocation_nominal', 'expected_profit']])

    return df

def generate_pie_chart(df):
    df_filtered = df[df['allocation_percent'] > 0].copy()
    if df_filtered.empty:
        return None

    plt.figure(figsize=(8, 8))
    plt.pie(df_filtered['allocation_percent'], labels=df_filtered['ticker'], autopct='%1.1f%%')
    plt.title('Distribusi Portofolio Saham')
    plt.tight_layout()
    pie_chart_path = os.path.join('static', 'portfolio_pie.png')
    plt.savefig(pie_chart_path)
    plt.close()
    return pie_chart_path

def save_prediction_chart(ticker, df, future_predictions):
    os.makedirs('static/chart_13saham', exist_ok=True)
    output_path = os.path.join('static', 'chart_13saham', f'prediksi_chart_{ticker}.png')
    df_plot = df[-30:].copy()
    df_plot_dates = df_plot.index.strftime('%Y-%m-%d').tolist()

    future_dates = []
    future_date = df.index[-1]
    for _ in range(len(future_predictions)):
        future_date += timedelta(days=1)
        while future_date.weekday() >= 5:
            future_date += timedelta(days=1)
        future_dates.append(future_date.strftime('%Y-%m-%d'))

    extended_values = df_plot['Close'].tolist() + [float(p) for p in future_predictions]
    split_index = len(df_plot_dates) - 1

    actual_start = pd.to_datetime(df_plot_dates[0]).strftime('%b %Y')
    actual_end = pd.to_datetime(df_plot_dates[-1]).strftime('%b %Y')
    pred_start = pd.to_datetime(future_dates[0]).strftime('%b %Y')
    pred_end = pd.to_datetime(future_dates[-1]).strftime('%b %Y')
    label_keterangan = f'Aktual: {actual_start}–{actual_end}, Prediksi: {pred_start}–{pred_end}'

    plt.figure(figsize=(12, 5))
    plt.plot(range(len(extended_values)), extended_values, label='Harga', marker='o', color='blue')
    plt.axvline(x=split_index, color='red', linestyle='--', label='Mulai Prediksi')
    plt.fill_between(range(split_index+1), min(extended_values), max(extended_values), color='gray', alpha=0.1, label='Aktual')
    plt.fill_between(range(split_index, len(extended_values)), min(extended_values), max(extended_values), color='orange', alpha=0.1, label='Prediksi')
    
    plt.title(f'Harga Penutupan {ticker}: Aktual vs Prediksi 3 Bulan')
    plt.xlabel(label_keterangan)
    plt.ylabel('Harga Penutupan')
    plt.xticks([])
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    return output_path

def format_df_for_display(df):
    df = df.copy()
    df['allocation_nominal'] = df['allocation_nominal'].apply(lambda x: f"IDR {x:,.0f}")
    df['current_price'] = df['current_price'].apply(lambda x: f"{x:.2f}")
    df['predicted_price'] = df['predicted_price'].apply(lambda x: f"{x:.2f}")
    df['predicted_price_3mo'] = df['predicted_price_3mo'].apply(lambda x: f"{x:.2f}")
    df['allocation_percent'] = df['allocation_percent'].apply(lambda x: f"{x * 100:.2f}%")
    df['expected_profit'] = df['expected_profit'].apply(lambda x: f"IDR {x:,.0f}")
    
    df = df.rename(columns={
        'ticker': 'Saham',
        'current_price': 'Harga Saat Ini',
        'predicted_price': 'Prediksi Besok',
        'predicted_price_3mo': 'Prediksi 3 Bulan',
        'allocation_percent': 'Bobot Portofolio (%)',
        'allocation_nominal': 'Alokasi (IDR)',
        'expected_profit': 'Estimasi Profit (IDR)',
    })

    return df[['Saham', 'Harga Saat Ini', 'Prediksi Besok', 'Prediksi 3 Bulan',
               'Bobot Portofolio (%)', 'Alokasi (IDR)', 'Estimasi Profit (IDR)']]

@app.route('/evaluasi', methods=['GET'])
def evaluate_portfolio_route():
    if 'results_cache_key' not in session:
        return jsonify({
            'message': 'Tidak ada hasil portofolio untuk dievaluasi.',
            'status': 'error'
        }), 400

    try:
        results = cache.load(session['results_cache_key'])
        if not results:
            return jsonify({
                'message': 'Data cache tidak ditemukan.',
                'status': 'error'
            }), 400

        df = pd.DataFrame(results)
        evaluasi = evaluasi_portfolio(df)
        return jsonify({
            'expected_return': round(evaluasi['expected_return'], 6),
            'expected_return_percent': round(evaluasi['expected_return'] * 100, 2),
            'risk': round(evaluasi['risk'], 6),
            'risk_percent': round(evaluasi['risk'] * 100, 4),
            'sharpe_ratio': round(evaluasi['sharpe_ratio'], 4),
            'num_assets': evaluasi['num_assets'],
            'status': 'success'
        })
    except Exception as e:
        return jsonify({
            'message': f'Terjadi kesalahan saat evaluasi: {str(e)}',
            'status': 'error'
        }), 500

if __name__ == '__main__':
    app.run(debug=True)