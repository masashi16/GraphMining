import argparse
import numpy as np
import networkx as nx
import node2vec
from gensim.models import Word2Vec


def parse_args():
    '''
    Parses the node2vec arguments.
    '''
    parser = argparse.ArgumentParser(description="Run node2vec.")

    # インプットデータ
    parser.add_argument('--input', nargs='?', default='graph/karate.edgelist',
                        help='Input graph path')
    # アウトプットデータ
    parser.add_argument('--output', nargs='?', default='emb/karate.emb',
                        help='Embeddings path')
    # 特徴空間の次元数
    parser.add_argument('--dimensions', type=int, default=128,
                        help='Number of dimensions. Default is 128.')
    # 1回のwalkの系列の長さ
    parser.add_argument('--walk-length', type=int, default=80,
                        help='Length of walk per source. Default is 80.')
    # ソースを決めた場合のwalkの回数
    parser.add_argument('--num-walks', type=int, default=10,
                        help='Number of walks per source. Default is 10.')
    # window size
    parser.add_argument('--window-size', type=int, default=10,
                        help='Context size for optimization. Default is 10.')

    parser.add_argument('--iter', default=1, type=int,
                        help='Number of epochs in SGD')

    parser.add_argument('--workers', type=int, default=8,
                        help='Number of parallel workers. Default is 8.')

    parser.add_argument('--p', type=float, default=1,
                        help='Return hyperparameter. Default is 1.')

    parser.add_argument('--q', type=float, default=1,
                        help='Inout hyperparameter. Default is 1.')

    # 引数に含めると，重み付きグラフ
    parser.add_argument('--weighted', dest='weighted', action='store_true',
                        help='Boolean specifying (un)weighted. Default is unweighted.')
    # 引数に含めると，重みなしグラフ
    parser.add_argument('--unweighted', dest='unweighted', action='store_false')
    parser.set_defaults(weighted=False)  # デフォルトで重みづけなし

    # 引数に含めると，向き付けありグラフ
    parser.add_argument('--directed', dest='directed', action='store_true',
                        help='Graph is (un)directed. Default is undirected.')
    # 引数に含めると，向き付けなしグラフ
    parser.add_argument('--undirected', dest='undirected', action='store_false')
    parser.set_defaults(directed=False)  # デフォルトで向き付けなし

    return parser.parse_args()


def read_graph():
    '''
    Reads the input network in networkx.
    '''
    if args.weighted:  # data : bool or list of (label,type) tuples (dictionary type)
        G = nx.read_edgelist(args.input, nodetype=int, data=(('weight',float),), create_using=nx.DiGraph())
    else:
        G = nx.read_edgelist(args.input, nodetype=int, create_using=nx.DiGraph())
        for edge in G.edges():
            G[edge[0]][edge[1]]['weight'] = 1  # unweightedは，重みを1に設定

    if not args.directed:
        G = G.to_undirected()

    return G


def learn_embeddings(walks):
    '''
    Learn embeddings by optimizing the Skipgram objective using SGD.
    '''
    walks = [list(map(str, walk)) for walk in walks]  # map(type, リスト)：リストの各要素をtypeに変換

    # sg: 1なら，skip-gram, 0なら，CBoW
    # min_count: n回未満登場する単語を破棄
    model = Word2Vec(walks, size=args.dimensions, window=args.window_size, min_count=0, sg=1, workers=args.workers, iter=args.iter)
    #model.wv.save_word2vec_format(args.output)
    model.save(args.output)

    return


def main(args):
    '''
    Pipeline for representational learning for all nodes in a graph.
    '''
    nx_G = read_graph()  # 引数でグラフデータを入力  --input ~.edgelist
    G = node2vec.Graph(nx_G, args.directed, args.p, args.q)  # node2vec.pyの中の，Graphクラスのインスタンスを生成
    G.preprocess_transition_probs()  # グラフのリンクの重みに従って，各ノードごとにその重みを反映した遷移確率でウォーク出来るように準備
    walks = G.simulate_walks(args.num_walks, args.walk_length)  # バイアス付きランダムウォークを開始
    learn_embeddings(walks)  # 上記で得られたノード系列データをインプットとして，skip-gramモデルで学習


if __name__ == "__main__":
    args = parse_args()
    main(args)
