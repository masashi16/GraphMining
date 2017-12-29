import numpy as np
import networkx as nx
import random

class Graph():
    def __init__(self, nx_G, is_directed, p, q):
        self.G = nx_G
        self.is_directed = is_directed
        self.p = p
        self.q = q


    def get_alias_edge(self, source, destination):
        # バイアス p, q を用いてバイアス付きの重みをエッジに割り振る
        # 入力は，エッジの端と端

        G = self.G
        p = self.p
        q = self.q

        unnormalized_probs = []
        for dst_nbr in sorted(G.neighbors(destination)):
            if dst_nbr == source:  # ソース(根元)に戻る確率を1/p倍 (ソースとの距離が0の場合)
                unnormalized_probs.append(G[destination][dst_nbr]['weight']/p)
            elif G.has_edge(dst_nbr, source):  # ソースとエッジを持つ ⇄ 距離が１ゆえ，バイアスなし
                unnormalized_probs.append(G[destination][dst_nbr]['weight'])
            else:  # それ以外は，ソースとの距離が2の場合，1/q倍
                unnormalized_probs.append(G[destination][dst_nbr]['weight']/q)
        norm_const = sum(unnormalized_probs)
        normalized_probs =  [float(u_prob)/norm_const for u_prob in unnormalized_probs]

        return alias_setup(normalized_probs)  # バイアス付きの重みエッジに対応した(J,q)



    def preprocess_transition_probs(self):
        # グラフのリンクの重みに従って，各ノードごとにその重みを反映した遷移確率でウォーク出来るように準備

        G = self.G
        is_directed = self.is_directed

        # alias_nodes: 各ノードごとの(J,q)を要素に入れる (これをalias_draw(J,q)に入れれば，重みに基づいた抽出ができる)
        alias_nodes = {}
        for node in G.nodes():
            unnormalized_probs = [G[node][nbr]['weight'] for nbr in sorted(G.neighbors(node))] # ノードのNNが持つ重みをリストに
            norm_const = sum(unnormalized_probs)
            normalized_probs = [float(u_prob)/norm_const for u_prob in unnormalized_probs]
            alias_nodes[node] = alias_setup(normalized_probs)

        # alias_edges: バイアス p,q の重み付けしたエッジに基づく(J,q)
        alias_edges = {}
        if is_directed:
            for edge in G.edges():
                alias_edges[edge] = self.get_alias_edge(edge[0], edge[1])
        else:
            for edge in G.edges():
                alias_edges[edge] = self.get_alias_edge(edge[0], edge[1])
                alias_edges[(edge[1], edge[0])] = self.get_alias_edge(edge[1], edge[0])

        self.alias_nodes = alias_nodes
        self.alias_edges = alias_edges

        return


    def node2vec_walk(self, walk_length, start_node):
        # スタートノードから，ランダムウォークを走らせ，ノード列を得る

        G = self.G
        alias_nodes = self.alias_nodes  # これは何？
        alias_edges = self.alias_edges

        walk = [start_node]  # ランダムウォークで得られるノード列

        # 1つずつランダムウォークを行っていく
        while len(walk) < walk_length:
            cur = walk[-1]  # 現在地のノード
            cur_nbrs = sorted(G.neighbors(cur))  # 現在のノードに接しているノードリスト

            if len(cur_nbrs) > 0:
                if len(walk) == 1:  # 一番初めだけは，バイアス付かないので特別
                    # バイアスなしでランダム抽出されたNNを加える
                    walk.append(cur_nbrs[alias_draw(alias_nodes[cur][0], alias_nodes[cur][1])])
                else:
                    # nextは，バイアス付きで選ばれたNNが加わる
                    prev = walk[-2]
                    next = cur_nbrs[alias_draw(alias_edges[(prev, cur)][0],
                                                alias_edges[(prev, cur)][1])]
                    walk.append(next)
            else:
                break

        return walk



    def simulate_walks(self, num_walks, walk_length):
        '''
        Repeatedly simulate random walks from each node.
        '''
        G = self.G
        walks = []
        nodes = list(G.nodes())
        print("Walk iteration:")
        for walk_iter in range(num_walks):
            print(str(walk_iter+1), '/', str(num_walks))
            random.shuffle(nodes)  # 毎回ランダムに始点のノードを選ぶ
            for node in nodes:
                walks.append(self.node2vec_walk(walk_length=walk_length, start_node=node))

        return walks




### Walker's Alias Method (復元抽出の高速アルゴリズム) ###

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
