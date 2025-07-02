import pandas as pd
import numpy as np
import pingouin as pg
# --- 使い方 ---
# 偏相関係数とは見かけの相関係数から隠れ変数の影響を除いた時の相関係数
# XとYには偏相関係数を求めたいデータを入力し、Zには隠れ変数(交絡変数)のデータを入力
# 偏相関係数は第３の変数を統計的に取り除いた後の関係性を示す数値であって変換後の散布図は描けないのでグラフはない。

# --- サンプルデータの作成 ---
# 例：学習時間(X)とテストの点数(Y)の相関を、IQ(Z)の影響を除外して見たい場合
# データを作成します。
# 学習時間(X)とIQ(Z)がテストの点数(Y)に影響を与えていると仮定します。
np.random.seed(42)
n = 100
iq_z = np.random.normal(100, 15, n)
study_time_x = np.random.normal(10, 2, n) + (iq_z - 100) / 5 # IQが高いほど勉強時間も少し長い傾向
test_score_y = 70 + 0.8 * study_time_x + 0.5 * (iq_z - 100) + np.random.normal(0, 5, n) # 点数は勉強時間とIQに依存

df = pd.DataFrame({
    'X_StudyTime': study_time_x,
    'Y_TestScore': test_score_y,
    'Z_IQ': iq_z
})

print("--- サンプルデータ (最初の5行) ---")
print(df.head())
print("-" * 30)

# --- まず、通常のピアソン相関係数を確認 ---
print("\n--- 通常のピアソン相関係数 ---")
# Z_IQの影響が含まれているため、XとYの相関は高く見える可能性がある
print(df.corr(method='pearson'))
print("-" * 30)

# --- 方法1: pingouinライブラリを使用 (簡単で推奨) ---
# pingouinがインストールされていない場合はpingouin をインストールしてください
print("\n--- 1. pingouin を使用した偏相関係数の計算 ---")

# 'Y_TestScore'と'X_StudyTime'の相関を、'Z_IQ'の影響を除外して計算
# x: 目的変数1, y: 目的変数2, covar: 統制変数（影響を取り除く変数）

print(partial_corr_pg)
print("\n解説:")
print(f"IQの影響を除外すると、学習時間とテストスコアの偏相関係数は r = {partial_corr_pg['r']['pearson']:.4f} となります。")
print("\nCI95% これは「もし同じ調査を100回繰り返したら、そのうち95回は、計算される偏相関係数がこの区間に入るでしょう」という統計的な推定の幅")
print("\np値は観測された相関が、偶然そうなっただけである確率")
print("\n一般的にこの値が0.05（5%）より小さいと、「偶然とは考えにくく、統計的に意味のある差（有意差あり）」と判断")
print("-" * 30)



# --- 方法2: NumPyと逆共分散行列を使用して手動で計算 ---
print("\n--- 2. NumPy を使用した偏相関係数の計算 ---")
print("この方法は、計算の仕組みを理解するのに役立ちます。")

def calculate_partial_correlation_manual(df, var1, var2, control_var):
    """
    NumPyを使い、逆共分散行列から偏相関係数を計算します。
    
    Args:
        df (pd.DataFrame): データフレーム
        var1 (str): 変数1の名前
        var2 (str): 変数2の名前
        control_var (str): 統制変数（影響を取り除く変数）の名前
        
    Returns:
        float: 偏相関係数
    """
    # 対象となる3つの変数のデータを選択
    data_subset = df[[var1, var2, control_var]]
    
    # 共分散行列を計算
    cov_matrix = data_subset.cov()
    
    # 逆共分散行列 (Precision Matrix) を計算
    try:
        precision_matrix = np.linalg.inv(cov_matrix.values)
    except np.linalg.LinAlgError:
        print("エラー: 共分散行列が特異であり、逆行列を計算できません。")
        return None
        
    # 逆共分散行列から偏相関係数を計算する
    # r(ij|k) = -P_ij / sqrt(P_ii * P_jj)
    # ここで P は逆共分散行列の要素
    p_ij = precision_matrix[0, 1]
    p_ii = precision_matrix[0, 0]
    p_jj = precision_matrix[1, 1]
    
    partial_r = -p_ij / np.sqrt(p_ii * p_jj)
    
    return partial_r

# 手動で計算を実行
partial_r_manual = calculate_partial_correlation_manual(df, 'X_StudyTime', 'Y_TestScore', 'Z_IQ')

if partial_r_manual is not None:
    print(f"\n学習時間(X)とテストスコア(Y)の偏相関係数（IQ(Z)を統制）:")
    print(f"計算結果 (NumPy): {partial_r_manual:.4f}")

    # pingouinの結果と比較
    print(f"計算結果 (pingouin): {partial_corr_pg['r']['pearson']:.4f}")
    print("2つの方法でほぼ同じ結果が得られることがわかる")

