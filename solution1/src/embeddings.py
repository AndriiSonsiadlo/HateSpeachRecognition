import numpy as np
from nltk.tokenize import TweetTokenizer
from main import create_vocabulary, tokenize_tweets
from tensorflow.keras.preprocessing.text import Tokenizer



class Embeddings():
    """
    A class to read the word embedding file and to create the word embedding matrix
    """

    def __init__(self, path, vector_dimension):
        self.path = path
        self.vector_dimension = vector_dimension

    @staticmethod
    def get_coefs(word, *arr):
        return word, np.asarray(arr, dtype='float32')

    def get_embedding_index(self):
        print("opening file")
        embeddings_index = dict(self.get_coefs(*o.split(" ")) for o in open(self.path, errors='ignore'))
        return embeddings_index

    def create_embedding_matrix(self, max_features, tokenized_tweets, tokenizer):
        """
        A method to create the embedding matrix
        """
        # all tokens and their embeddings dictionairy. ~1193514 tokens
        model_embed = self.get_embedding_index()
        embedding_matrix = np.zeros((max_features + 1, self.vector_dimension))

        if isinstance(tokenizer, TweetTokenizer):
            # all tokens and their embeddings dictionairy. ~1193514 tokens
            vocab = create_vocabulary(tokenized_tweets)


            print(f"vocab len: {len(vocab)}")

            for i, (word, index) in enumerate(vocab.items()):
                if i > max_features:
                    break
                else:
                    try:
                        embedding_vector = model_embed[word]
                        if embedding_vector is not None:
                            embedding_matrix[index] = embedding_vector
                    except:
                        continue

            return embedding_matrix
        else:
            print(f'vocab size={len(tokenizer.word_index.items())}')
            for word, index in Tokenizer(tokenizer).word_index.items():
                if index > max_features:
                    break
                else:
                    try:
                        embedding_matrix[index] = model_embed[word]
                    except:
                        continue
            return embedding_matrix




