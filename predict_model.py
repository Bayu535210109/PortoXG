# predict_model.py

import matplotlib.pyplot as plt
import numpy as np
import os

def prediksi_dan_optimasi(dana_input):
    saham = ['9988.HK', 'SONY', '000660.KQ', '005930.KS', 'TSLA',
             '1810.HK', 'GOOGL', 'BABA', 'AAPL', 'AMZN']
    porsi = np.array([0.18, 0.15, 0.13, 0.11, 0.09, 0.08, 0.07, 0.07, 0.06, 0.06])
    dana_saham = dana_input * porsi
    hasil = dict(zip(saham, dana_saham))

    # Simpan pie chart ke folder static
    fig, ax = plt.subplots()
    ax.pie(dana_saham, labels=saham, autopct='%1.1f%%', startangle=90)
    ax.set_title('Alokasi Dana Prediksi')
    plt.tight_layout()
    os.makedirs("static", exist_ok=True)
    plt.savefig("static/grafik_pie.png")
    plt.close()

    return hasil
