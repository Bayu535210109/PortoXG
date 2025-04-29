import pandas as pd

def predict_all_stocks(tickers, process_fn):
    results = []
    for ticker in tickers:
        try:
            res = process_fn(ticker)
            if res and res.get("next_close_prediction") and res.get("last_close"):
                pred = res["next_close_prediction"]
                last = res["last_close"]
                pred_return = (pred - last) / last
                results.append({
                    "ticker": ticker,
                    "last_price": last,
                    "pred_price": pred,
                    "pred_return": pred_return
                })
        except Exception as e:
            print(f"❌ Gagal prediksi {ticker}: {e}")
            continue
    return pd.DataFrame(results)

def optimize_portfolio(df_pred, total_dana, excluded=[]):
    df_use = df_pred[~df_pred['ticker'].isin(excluded)].copy()
    if df_use.empty:
        raise ValueError("Semua saham dikecualikan!")
    total_return = df_use['pred_return'].sum()
    df_use['weight'] = df_use['pred_return'] / total_return
    df_use['allocation'] = df_use['weight'] * total_dana
    return df_use[['ticker', 'last_price', 'pred_price', 'pred_return', 'weight', 'allocation']]