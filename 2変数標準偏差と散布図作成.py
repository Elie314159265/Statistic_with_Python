import numpy as np
from scipy import stats
import pandas as pd
import math
import matplotlib.pyplot as plt #グラフ描画のためのライブライ
import japanize_matplotlib # 日本語ラベルのためのライブラリ

def calculate_pearson_manual(x, y):
    """
    Pythonの標準ライブラリのみを使用してピアソンの積率相関係数を計算
    
    Args:
        x (list): 1つ目のデータセット
        y (list): 2つ目のデータセット
        
    Returns:
        float: ピアソンの積率相関係数。計算不可能な場合はNoneを返す
    """
    n = len(x)
    if n != len(y) or n == 0:
        print("エラー: 2つのリストは同じ長さで、空であってはならない")
        return None

    mean_x = sum(x) / n
    mean_y = sum(y) / n
    covariance = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
    std_dev_x = math.sqrt(sum((xi - mean_x)**2 for xi in x))
    std_dev_y = math.sqrt(sum((yi - mean_y)**2 for yi in y))

    if std_dev_x == 0 or std_dev_y == 0:
        print("エラー: 少なくとも一方のリストの標準偏差が0。相関係数は計算できない")
        return None

    correlation = covariance / (std_dev_x * std_dev_y)
    return correlation

# --- 散布図を作成する関数 ---
def create_scatter_plot(x, y, title, corr_val):
    """指定されたデータで散布図を作成して表示"""
    """Y軸のラベルは適切に変更"""
    plt.figure(figsize=(8, 6))
    plt.scatter(x, y, alpha=0.7, edgecolors='b', s=100)
    plt.title(f'{title}\n相関係数: {corr_val:.4f}', fontsize=16)
    plt.xlabel('Xデータ', fontsize=12)
    plt.ylabel('Y・Z・Wデータ', fontsize=12)
    plt.grid(True)
    plt.show()

# --- サンプルデータ ---
data_x = [1, 2, 3, 4, 5, 6, 7, 8, 9]
data_y = [2, 3, 4, 5, 6, 7, 8, 9, 10] # data_x と強い正の相関
data_z = [9, 8, 7, 6, 5, 4, 3, 2, 1] # data_x と強い負の相関
data_w = [5, 1, 8, 3, 9, 2, 6, 4, 7] # data_x と相関が低い

print("--- サンプルデータ ---")
print(f"X: {data_x}")
print(f"Y: {data_y}")
print(f"Z: {data_z}")
print(f"W: {data_w}")
print("-" * 20)

# --- 方法1: NumPyを使用 (Numpyはよく利用される) ---
print("\n--- 1. NumPyを使用 ---")
corr_xy_numpy = np.corrcoef(data_x, data_y)[0, 1]
print(f"XとYの相関係数 (NumPy): {corr_xy_numpy:.4f}")
corr_xz_numpy = np.corrcoef(data_x, data_z)[0, 1]
print(f"XとZの相関係数 (NumPy): {corr_xz_numpy:.4f}")
corr_xw_numpy = np.corrcoef(data_x, data_w)[0, 1]
print(f"XとWの相関係数 (NumPy): {corr_xw_numpy:.4f}")

# --- 方法2: SciPyを使用 (p値も計算) ---
print("\n--- 2. SciPyを使用 ---")
corr_xy_scipy, p_value_xy = stats.pearsonr(data_x, data_y)
print(f"XとYの相関係数 (SciPy): {corr_xy_scipy:.4f}, p値: {p_value_xy:.4f}")
corr_xz_scipy, p_value_xz = stats.pearsonr(data_x, data_z)
print(f"XとZの相関係数 (SciPy): {corr_xz_scipy:.4f}, p値: {p_value_xz:.4f}")
corr_xw_scipy, p_value_xw = stats.pearsonr(data_x, data_w)
print(f"XとWの相関係数 (SciPy): {corr_xw_scipy:.4f}, p値: {p_value_xw:.4f}")

# --- 方法3: Pandasを使用 (データフレーム形式の場合に便利) ---
print("\n--- 3. Pandasを使用 ---")
df = pd.DataFrame({'X': data_x, 'Y': data_y, 'Z': data_z, 'W': data_w})
corr_matrix_pandas = df.corr(method='pearson')
print("Pandasによる相関行列:")
print(corr_matrix_pandas)

# --- 方法4: Python標準ライブラリのみで計算 ---
print("\n--- 4. Python標準ライブラリのみ ---")
corr_xy_manual = calculate_pearson_manual(data_x, data_y)
print(f"XとYの相関係数 (手動計算): {corr_xy_manual:.4f}")
corr_xz_manual = calculate_pearson_manual(data_x, data_z)
print(f"XとZの相関係数 (手動計算): {corr_xz_manual:.4f}")
corr_xw_manual = calculate_pearson_manual(data_x, data_w)
print(f"XとWの相関係数 (手動計算): {corr_xw_manual:.4f}")

# --- 5. 散布図の作成 ---
print("\n--- 5. 散布図の作成 (Matplotlib) ---")
# グラフの日本語フォント設定
# japanize_matplotlib がない場合は、コメントアウトしてください
# pip install japanize-matplotlib
plt.rcParams['font.family'] = 'IPAexGothic'

# XとYの散布図
create_scatter_plot(data_x, data_y, 'XとYの散布図 (強い正の相関)', corr_xy_scipy)

# XとZの散布図
create_scatter_plot(data_x, data_z, 'XとZの散布図 (強い負の相関)', corr_xz_scipy)

# XとWの散布図
create_scatter_plot(data_x, data_w, 'XとWの散布図 (相関が低い)', corr_xw_scipy)
