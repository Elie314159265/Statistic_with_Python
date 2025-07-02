#ケンドールの順位相関係数基本概念
#あるペア(i, j)について、変数Xでも変数Yでも順位の大小関係が同じ方向（例：X_i < X_j かつ Y_i < Y_j）なら協和ペアと呼ぶ
#例えばX_i=1,X_j=2かつY_i=2,Y_j=3なら協和ペア
#大小関係が逆方向（例：X_i < X_j だが Y_i > Y_j）なら不協和ペアと呼ぶ
#例えばX_i=1,X_j=2かつY_i=2,Y_j=1なら不協和ペア
#ケンドールの順位相関係数は、基本的に「(協和ペアの数 - 不協和ペアの数) / (総ペア数)」で計算されます

from scipy import stats  #同順位（タイ）がある場合も考慮して自動でけいさんが行えるライブラリ
from itertools import combinations #データのすべてのペアを効率的に生成。各ペアについて、協和か不協和かを判断し、それぞれのカウンターを増やす
import numpy as np

def calculate_kendall_tau_manual(x, y):
    """
    手動でケンドールの順位相関係数（tau-a）を計算します。
    
    Args:
        x (list): 1つ目のデータセット (数値のリスト)
        y (list): 2つ目のデータセット (数値のリスト)
        
    Returns:
        float: ケンドールのタウ。計算不可能な場合はNone。
    """
    n = len(x)
    if n != len(y) or n < 2:
        print("エラー: 2つのリストは同じ長さで、要素が2つ以上必要です。")
        return None

    concordant_pairs = 0  # 協和ペア（順位が同じ方向）
    discordant_pairs = 0 # 不協和ペア（順位が逆の方向）

    """ combinationsの説明
    combinations( [0, 1, 2, 3, 4], 2) を実行すると、以下のようなペアが順番に生成されます。
    (0, 1)(0, 2)(0, 3)(0, 4)(1, 2)(1, 3)(1, 4)(2, 3)(2, 4)(3, 4)
    (1, 0) のような逆の組み合わせや、(0, 0) のような同じもの同士の組み合わせは含まれない
    """
    # 全てのペア (i, j) where i < j についてループ
    for i, j in combinations(range(n), 2):
        # (x[i] - x[j]) と (y[i] - y[j]) の符号を比較
        x_diff_sign = np.sign(x[i] - x[j])
        y_diff_sign = np.sign(y[i] - y[j])
        
        # 同順位（タイ）がない場合
        if x_diff_sign != 0 and y_diff_sign != 0:
            # 協和ペア
            if x_diff_sign == y_diff_sign:
                concordant_pairs += 1
            #不協和ペア
            else:
                discordant_pairs += 1
    
    # ケンドールのタウ (tau-a) の計算式
    # (協和ペア数 - 不協和ペア数) / (総ペア数)
    # 総ペア数 = n * (n - 1) / 2
    tau = (concordant_pairs - discordant_pairs) / (n * (n - 1) / 2)
    
    return tau

# --- サンプルデータ ---
# 順位データや、正規分布に従わない可能性のあるデータ
# 例：5人の審査員がつけた2つの製品A, Bのランキング
product_a_rank = [1, 2, 3, 4, 5]
product_b_rank_positive = [1, 3, 2, 5, 4] # Aと似た傾向（正の相関）
product_b_rank_negative = [5, 4, 3, 2, 1] # Aと逆の傾向（負の相関）
product_b_rank_weak = [3, 1, 5, 2, 4]     # Aとあまり関係ない傾向（相関が低い）

print("--- サンプルデータ ---")
print(f"製品Aの順位: {product_a_rank}")
print(f"製品Bの順位 (正の相関): {product_b_rank_positive}")
print(f"製品Bの順位 (負の相関): {product_b_rank_negative}")
print(f"製品Bの順位 (相関弱い): {product_b_rank_weak}")
print("-" * 30)

# --- 方法1: SciPyを使用 (簡単で推奨) ---
# stats.kendalltau は (相関係数, p値) のタプルを返す
print("\n--- 1. SciPy を使用したケンドールタウの計算 ---")

# 正の相関のケース
tau_pos, p_value_pos = stats.kendalltau(product_a_rank, product_b_rank_positive)
print(f"AとB(正)のタウ: {tau_pos:.4f}, p値: {p_value_pos:.4f}")

# 負の相関のケース
tau_neg, p_value_neg = stats.kendalltau(product_a_rank, product_b_rank_negative)
print(f"AとB(負)のタウ: {tau_neg:.4f}, p値: {p_value_neg:.4f}")

# 相関が弱いケース
tau_weak, p_value_weak = stats.kendalltau(product_a_rank, product_b_rank_weak)
print(f"AとB(弱)のタウ: {tau_weak:.4f}, p値: {p_value_weak:.4f}")
print("-" * 30)


# --- 方法2: 手動で計算 ---
# 注意: この手動計算は同順位（タイ）を考慮しない最も基本的なtau-aです。
# SciPyのkendalltauは同順位を考慮したtau-bを計算するため、タイがあると値が少し異なります。
print("\n--- 2. 手動での計算 (協和/不協和ペア) ---")

# 正の相関のケース
tau_pos_manual = calculate_kendall_tau_manual(product_a_rank, product_b_rank_positive)
print(f"AとB(正)のタウ (手動): {tau_pos_manual:.4f}")

# 負の相関のケース
tau_neg_manual = calculate_kendall_tau_manual(product_a_rank, product_b_rank_negative)
print(f"AとB(負)のタウ (手動): {tau_neg_manual:.4f}")

# 相関が弱いケース
tau_weak_manual = calculate_kendall_tau_manual(product_a_rank, product_b_rank_weak)
print(f"AとB(弱)のタウ (手動): {tau_weak_manual:.4f}")
print("\nSciPyの結果と一致することがわかります（サンプルデータに同順位がないため）。")
