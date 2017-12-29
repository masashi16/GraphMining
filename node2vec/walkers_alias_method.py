import numpy as np

l = [0.1, 0.05, 0.3, 0.1, 0.45]
l_name = ['A', 'B', 'C', 'D', 'E']

#print(l)
#print(ll)

# 確率の重みのリストを渡すと，
# J:どのブロックにどれで補ったかの情報が書かれたリスト，q:各ブロックの前半部に含まれる確率
def alias_setup(probs):
    n = len(probs)
    q = np.zeros(n)  # 各要素の確率をリストの長さ倍したものを格納するもの
    J = np.zeros(n, dtype=np.int)

    smaller = []
    larger = []
    for i, prob in enumerate(probs):
        q[i] = n * prob
        if q[i] < 1.0:
            smaller.append(i)
        else:
            larger.append(i)

    while len(smaller) > 0 and len(larger) > 0:
        small = smaller.pop(-1)  # リストsmallerの末尾の要素を削除し，smallには削除された末尾が入る
        large = larger.pop(-1)

        # 1より大きいものと，1より小さいものを一対一のペアで割り振りを考えていく：
        J[small] = large  # どこにどれを割り振ったかを記録
        q[large] = q[large] - (1.0 - q[small])  # 1よりデカかったものを，小さいやつに振り分けてく，その余りを入れる

        # 割り振った後の，大きいやつの余りが1より大きいかどうか：
        if q[large] < 1.0:
            smaller.append(large)  # 1より小さくなったら，もう振り分けないので，新たなブロックとして加える
        else:
            larger.append(large)  # 1より大きかったら，また1より大きいので，割り振るリストとして加える

    return J, q

# 上記で作った J,qを使って，１回抽出（ ⇨ これを繰り返せば復元抽出）
def alias_draw(J, q):
    n  = len(J)

    k = int(np.floor(np.random.rand() * n))  # 一様に「0 ~ n-1」のどれかの整数を返す (np.floorは切捨て)

    if np.random.rand() < q[k]:  # q[k]は1より小さい確率で必ずブロックの始めに入るので，それより小さかったら，k
        return k
    else:  # 大きかったら，k番目の中の補ったやつの番号，J[k]
        return J[k]


"""
J, q = alias_setup(l)

# Generate variates.
X = np.zeros(10)
for nn in range(10):
    X[nn] = alias_draw(J, q)
print(X)
"""
