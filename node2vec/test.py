from gensim.models import word2vec

model = word2vec.Word2Vec.load('emb/airport.emb')
word = model.most_similar(positive=["1"])
print(word)
