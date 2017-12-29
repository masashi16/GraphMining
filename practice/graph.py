import networkx as nx
import matplotlib.pyplot as plt

"""
G = nx.Graph()
G.add_node(1)
G.add_nodes_from([2,3])
print(G.nodes())
"""


#グラフを構築
G = nx.read_edgelist('karate.edgelist')

#ノード数とエッジ数を出力
print(nx.number_of_nodes(G))
print(nx.number_of_edges(G))

#Gのノード一覧
print(G.nodes())

#ノード22の隣接ノード
print(G.neighbors('22'))

#レイアウトの取得
pos = nx.spring_layout(G)

#pagerankの計算, Webページの重要度を決定するために使われる(Googleが特許,1998)
pr = nx.pagerank(G)
#print(pr)  #pagerankの各ノードのスコア


#可視化
#plt.figure(figsize=(6,6))
nx.draw_networkx_edges(G, pos)
#nx.draw_networkx_nodes(G, pos)
nx.draw_networkx_nodes(G, pos, node_color=list(pr.values()), cmap=plt.cm.Reds)
plt.axis('off')
plt.show()
