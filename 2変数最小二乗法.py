import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import japanize_matplotlib # 日本語ラベルのためのライブラリ

def calculate_least_squares_manual(x, y):
    """
    正規方程式を直接計算して、単回帰の係数を手動で求めます。
    
    Args:
        x (np.array): 説明変数のデータ
        y (np.array): 目的変数のデータ
        
    Returns:
        tuple: (切片 beta_0, 傾き beta_1)
    """
    # 計画行列Xを作成します。最初の列は切片項のための「1」
    # np.vstack([x, np.ones(len(x))]).T としても良いがここではより汎用的なc_を使う
    X = np.c_[np.ones(len(x)), x]
    
    # 正規方程式を解く: beta = (X^T * X)^-1 * X^T * y
    try:
        # (X^T * X)
        xtx = np.dot(X.T, X)
        # (X^T * X)^-1
        xtx_inv = np.linalg.inv(xtx)
        # (X^T * X)^-1 * X^T
        xtx_inv_xt = np.dot(xtx_inv, X.T)
        # beta = (X^T * X)^-1 * X^T * y
        beta = np.dot(xtx_inv_xt, y)
        
        beta_0 = beta[0] # 切片
        beta_1 = beta[1] # 傾き
        
        return beta_0, beta_1
        
    except np.linalg.LinAlgError:
        print("エラー: 行列計算中に問題が発生しました。逆行列が計算できない可能性があります。")
        return None, None

# --- 1. サンプルデータの作成 ---
# y = 2x + 5 の関係にノイズを加えたデータを作成
np.random.seed(0)
x_data = 2 * np.random.rand(100, 1)
y_data = 5 + 2 * x_data + np.random.randn(100, 1)

# データを1次元配列に変換
x_data_flat = x_data.flatten()
y_data_flat = y_data.flatten()

# --- 2. scikit-learn を使用した計算 (簡単で推奨) ---
print("--- 1. scikit-learn を使用 ---")
# モデルのインスタンスを作成
model = LinearRegression()
# モデルをデータに適合させる (学習)
# scikit-learnはXを2次元配列として要求するため、元のx_dataを使用
model.fit(x_data, y_data)

# 結果の取得
beta_0_sklearn = model.intercept_[0]
beta_1_sklearn = model.coef_[0][0]
print(f"切片 (β0): {beta_0_sklearn:.4f}")
print(f"傾き (β1): {beta_1_sklearn:.4f}")
print("-" * 30)

# --- 3. 正規方程式を手動で計算 ---
print("--- 2. 正規方程式を手動で計算 ---")
beta_0_manual, beta_1_manual = calculate_least_squares_manual(x_data_flat, y_data_flat)

if beta_0_manual is not None:
    print(f"切片 (β0): {beta_0_manual:.4f}")
    print(f"傾き (β1): {beta_1_manual:.4f}")
    print("scikit-learnの結果と一致することがわかります。")
print("-" * 30)

# --- 4. 結果の可視化 ---
print("--- 3. 結果をグラフにプロット ---")
# 予測値の計算
y_pred = beta_0_sklearn + beta_1_sklearn * x_data_flat

plt.figure(figsize=(10, 6))
# 元のデータの散布図
plt.scatter(x_data_flat, y_data_flat, alpha=0.7, label='実測値')
# 最小二乗法による回帰直線
plt.plot(x_data_flat, y_pred, color='red', linewidth=2, label='回帰直線 (y = {:.2f}x + {:.2f})'.format(beta_1_sklearn, beta_0_sklearn))

plt.title('最小二乗法による単回帰分析', fontsize=16)
plt.xlabel('説明変数 (x)', fontsize=12)
plt.ylabel('目的変数 (y)', fontsize=12)
plt.legend()
plt.grid(True)
plt.show()
