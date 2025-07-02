from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib # 日本語ラベルのためのライブラリ

def calculate_spearman_manual(x, y):
    """
    手動でスピアマンの順位相関係数を計算します。
    データを順位に変換し、その順位データでピアソンの積率相関係数を計算
    
    Args:
        x (list): 1つ目のデータセット
        y (list): 2つ目のデータセット
        
    Returns:
        float: スピアマンの相関係数。計算不可能な場合はNone。
    """
    n = len(x)
    if n != len(y) or n < 2:
        print("エラー: 2つのリストは同じ長さで、要素が2つ以上必要")
        return None
        
    rank_x = stats.rankdata(x)
    rank_y = stats.rankdata(y)
    rho = np.corrcoef(rank_x, rank_y)[0, 1]
    return rho

#量的変数をグラフにプロット
def create_scatter_plot(x, y, title, pearson_r, spearman_rho):
    """指定されたデータで散布図を作成して表示する"""
    plt.figure(figsize=(8, 6))
    plt.scatter(x, y, alpha=0.8, edgecolors='b', s=100)
    plt.title(title, fontsize=16,y=1.1)
    # グラフのサブタイトルとして、ピアソンとスピアマンの相関係数を両方表示
    plt.suptitle(f'\nピアソン r = {pearson_r:.4f}  |  スピアマン ρ = {spearman_rho:.4f}', fontsize=14, y=0.96)
    plt.xlabel('Xデータ', fontsize=12)
    plt.ylabel('Yデータ', fontsize=12)
    plt.grid(True)
    plt.show()

# --- サンプルデータ ---
# 例1: 単調増加だが、直線的ではない関係
x_nonlinear = [1, 2, 3, 4, 5, 6, 7, 8]
y_nonlinear = [1, 4, 9, 16, 25, 36, 49, 64] # y = x^2 の関係

# 例2: 外れ値を含むデータ
x_outlier = [1, 2, 3, 4, 5, 100]
y_outlier = [2, 4, 6, 8, 10, 1] # 最後の点だけが外れ値

print("--- サンプルデータ ---")
print(f"非線形データ X: {x_nonlinear}")
print(f"非線形データ Y: {y_nonlinear}")
print(f"外れ値を含むデータ X: {x_outlier}")
print(f"外れ値を含むデータ Y: {y_outlier}")
print("-" * 30)

# --- まず、比較のためにピアソンの相関係数を計算 ---
print("\n--- 比較: ピアソンの積率相関係数 ---")
pearson_nonlinear, _ = stats.pearsonr(x_nonlinear, y_nonlinear)
print(f"非線形データのピアソン相関: {pearson_nonlinear:.4f}")
pearson_outlier, _ = stats.pearsonr(x_outlier, y_outlier)
print(f"外れ値データのピアソン相関: {pearson_outlier:.4f}")
print("-" * 30)

# --- 方法1: SciPyを使用 (簡単で推奨) ---
print("\n--- 1. SciPy を使用したスピアマン相関の計算 ---")
rho_nonlinear, p_nonlinear = stats.spearmanr(x_nonlinear, y_nonlinear)
print(f"非線形データのスピアマン相関: {rho_nonlinear:.4f}, p値: {p_nonlinear:.4f}")
rho_outlier, p_outlier = stats.spearmanr(x_outlier, y_outlier)
print(f"外れ値データのスピアマン相関: {rho_outlier:.4f}, p値: {p_outlier:.4f}")
print("-" * 30)

# --- 方法2: 手動で計算 ---
print("\n--- 2. 手動での計算 (順位のピアソン相関) ---")
rho_nonlinear_manual = calculate_spearman_manual(x_nonlinear, y_nonlinear)
print(f"非線形データのスピアマン相関 (手動): {rho_nonlinear_manual:.4f}")
rho_outlier_manual = calculate_spearman_manual(x_outlier, y_outlier)
print(f"外れ値データのスピアマン相関 (手動): {rho_outlier_manual:.4f}")
print("-" * 30)

# --- 3. 生データの散布図を作成 ---
print("\n--- 3. 散布図の作成 (Matplotlib) ---")

# 非線形データの散布図
create_scatter_plot(x_nonlinear, y_nonlinear, '非線形な関係の散布図', pearson_nonlinear, rho_nonlinear)

# 外れ値を含むデータの散布図
create_scatter_plot(x_outlier, y_outlier, '外れ値を含むデータの散布図', pearson_outlier, rho_outlier)
