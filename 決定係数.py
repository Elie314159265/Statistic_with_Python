import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import japanize_matplotlib # 日本語ラベルのためのライブラリ

def calculate_least_squares_manual(x, y):
    """
    正規方程式を直接計算して、単回帰の係数を手動で求める
    """
    X = np.c_[np.ones(len(x)), x]
    try:
        beta = np.linalg.inv(X.T @ X) @ X.T @ y
        return beta[0], beta[1]
    except np.linalg.LinAlgError:
        print("エラー: 行列計算中に問題が発生しました")
        return None, None

def calculate_r2_manual(y_true, y_pred):
    """
    決定係数(R^2)を定義に基づいて手動で計算
    R^2 = 1 - (残差平方和 / 全平方和)
    """
    # 残差平方和 (SSE): 実測値と予測値の差の二乗和
    sse = np.sum((y_true - y_pred)**2)
    # 全平方和 (SST): 実測値と実測値の平均値の差の二乗和
    sst = np.sum((y_true - np.mean(y_true))**2)
    
    if sst == 0:
        # データにばらつきがない場合は計算不能
        return 1.0
        
    r2 = 1 - (sse / sst)
    return r2

# --- 1. CSVファイルからデータを読み込む ---
try:
    # --- ▼▼▼ 設定項目 ▼▼▼ ---
    file_path = 'sample_data.csv'
    x_column = 'x_value'
    y_column = 'y_value'
    # --- ▲▲▲ 設定項目 ▲▲▲ ---

    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"'{file_path}' が見つかりません。サンプルデータを作成します")
        np.random.seed(0)
        sample_x = 2 * np.random.rand(100, 1)
        sample_y = 5 + 2 * sample_x + np.random.randn(100, 1)
        df = pd.DataFrame({x_column: sample_x.flatten(), y_column: sample_y.flatten()})
        df.to_csv(file_path, index=False)
        print(f"'{file_path}' を作成しました")

    x_data = df[[x_column]].values
    y_data = df[[y_column]].values
    x_data_flat = x_data.flatten()
    y_data_flat = y_data.flatten()

except (FileNotFoundError, KeyError) as e:
    print(f"エラーが発生しました: {e}")
    exit()

# --- 2. scikit-learn を使用した計算 ---
print("--- 1. scikit-learn を使用 ---")
model = LinearRegression()
model.fit(x_data, y_data)

beta_0_sklearn = model.intercept_[0]
beta_1_sklearn = model.coef_[0][0]
# .score()メソッドで決定係数を計算
r2_sklearn = model.score(x_data, y_data)

print(f"切片 (β0): {beta_0_sklearn:.4f}")
print(f"傾き (β1): {beta_1_sklearn:.4f}")
print(f"決定係数 (R^2): {r2_sklearn:.4f}")
print("-" * 30)

# --- 3. 手動での計算 ---
print("--- 2. 手動での計算 ---")
beta_0_manual, beta_1_manual = calculate_least_squares_manual(x_data_flat, y_data_flat)

if beta_0_manual is not None:
    # 手動で計算した係数から予測値を計算
    y_pred_manual = beta_0_manual + beta_1_manual * x_data_flat
    # 決定係数を手動で計算
    r2_manual = calculate_r2_manual(y_data_flat, y_pred_manual)
    
    print(f"切片 (β0): {beta_0_manual:.4f}")
    print(f"傾き (β1): {beta_1_manual:.4f}")
    print(f"決定係数 (R^2): {r2_manual:.4f}")
    print("scikit-learnの結果と一致することがわかる")
print("-" * 30)

# --- 4. 結果の可視化 ---
print("--- 3. 結果をグラフにプロット ---")
y_pred_sklearn = model.predict(x_data)

plt.figure(figsize=(10, 6))
plt.scatter(x_data_flat, y_data_flat, alpha=0.7, label='実測値')
# 凡例に決定係数を追加
plt.plot(x_data_flat, y_pred_sklearn, color='red', linewidth=2, 
         label=f'回帰直線 (y = {beta_1_sklearn:.2f}x + {beta_0_sklearn:.2f})\n決定係数 R² = {r2_sklearn:.4f}')

plt.title('最小二乗法による単回帰分析', fontsize=16)
plt.xlabel(f'説明変数 ({x_column})', fontsize=12)
plt.ylabel(f'目的変数 ({y_column})', fontsize=12)
plt.legend()
plt.grid(True)
plt.show()
