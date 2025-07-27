# run once, transform GloVe -> Word2Vec. 
# later can directly use the Word2Vec.txt file to do. 

from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import KeyedVectors

# modify them as the REAL path!!
glove_input_file = "glove6B/glove.6B.300d.txt"
word2vec_output_file = "glove6B/glove.6B.300d.word2vec.txt"

glove2word2vec(glove_input_file, word2vec_output_file)
