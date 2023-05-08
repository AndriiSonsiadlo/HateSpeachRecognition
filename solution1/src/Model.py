import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Embedding, concatenate, Dropout, concatenate
from tensorflow.keras.layers import Bidirectional


class LSTMmodel():
    """
    A recurrent neural network for hate-speech analysis

    """

    def __init__(self, embedding_matrix, embedding_dim, max_len, X_additional=None):
        inp1 = Input(shape=(max_len,))
        x = Embedding(embedding_matrix.shape[0], embedding_dim, weights=[embedding_matrix])(inp1)
        x = Bidirectional(LSTM(256, return_sequences=True))(x)
        x = Bidirectional(LSTM(150))(x)
        x = Dense(128, activation="relu")(x)
        x = Dropout(0.1)(x)
        x = Dense(64, activation="relu")(x)
        x = Dense(3, activation='softmax')(x)
        model = Model(inputs=inp1, outputs=x)

        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
        self.model = model