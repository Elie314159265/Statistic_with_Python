import pandas as pd
import numpy as np
import statsmodels.api as sm

# --- サンプル時系列データの作成 ---
# 例: 緩やかに上昇し、周期性を持つような架空の月次売上データ
np.random.seed(42)
# データ数は100件
time = np.arange(100)
# トレンド成分 + 季節性成分 + ノイズ
sales_data = 100 + time * 1.5 + 20 * np.sin(time * np.pi / 6) + np.random.normal(0, 5, 100)
ts = pd.Series(sales_data)

print("--- サンプル時系列データ ---")
print(ts)
print("-" * 40)

# --- 方法1: Pandas を使用 ---
# Seriesオブジェクトの .autocorr() メソッドを使用
# 引数 lag=1 を指定することで、ラグ1の自己相関係数を直接計算可能
print("\n--- 1. Pandas を使用 ---")
autocorr_pandas = ts.autocorr(lag=1)
print(f"ラグ1の自己相関係数 (Pandas): {autocorr_pandas:.4f}")
print("-" * 40)


# --- 方法2: NumPy を使用 ---
# 元のデータと、1つずらしたデータとのピアソン相関係数を計算
# これが自己相関係数の定義そのもの
print("\n--- 2. NumPy を使用 ---")
# 元のデータ (t=1 から t=n まで)
original_series = ts[1:]
# 1つ前のデータ (t=0 から t=n-1 まで)
shifted_series = ts[:-1]

# np.corrcoefは相関行列を返すため、[0, 1]の要素を取得
autocorr_numpy = np.corrcoef(original_series, shifted_series)[0, 1]
print(f"ラグ1の自己相関係数 (NumPy): {autocorr_numpy:.4f}")
print("-" * 40)


# --- 方法3: Statsmodels を使用 ---
# statsmodels.tsa.stattools.acf (AutoCorrelation Function) を使用
# この関数は指定したラグまでの自己相関係数を配列で返す
print("\n--- 3. Statsmodels を使用 ---")
# nlagsにラグの最大数を指定。fft=Trueは高速な計算方法
# acf_values[0]はラグ0の自己相関（常に1）なので、[1]がラグ1の自己相関
acf_values = sm.tsa.acf(ts, nlags=1, fft=True)
autocorr_statsmodels = acf_values[1]
print(f"ラグ1の自己相関係数 (Statsmodels): {autocorr_statsmodels:.4f}")
print("-" * 40)

print("\n結論: どの方法でもほぼ同じ結果が得られます。")

# 参考: データのプロット
import matplotlib.pyplot as plt
import japanize_matplotlib

plt.figure(figsize=(10, 6))
plt.plot(ts, marker='o', linestyle='-')
plt.title('サンプル時系列データ（架空の月次売上）', fontsize=16)
plt.xlabel('時間', fontsize=12)
plt.ylabel('売上', fontsize=12)
plt.grid(True)
plt.show()
