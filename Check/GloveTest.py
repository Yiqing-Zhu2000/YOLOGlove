from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import KeyedVectors

word2vec_output_file = "glove6B/glove.6B.50d.word2vec.txt"
model = KeyedVectors.load_word2vec_format(word2vec_output_file, binary=False)

vector = model["king"]
print(vector.shape)  # (300,)

similarity = model.similarity("king", "queen")
print(similarity)  # 例如输出 0.72

print(model.most_similar("apple", topn=5))

result = model.most_similar(positive=['woman', 'king'], negative=['man'])
print(result[0])  # 输出 queen
