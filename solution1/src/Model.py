import tensorflow as tf
from keras.layers import BatchNormalization
from tensorflow import keras
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Embedding, concatenate, Dropout, concatenate, SpatialDropout1D, Bidirectional


class LSTMmodel():
    """
    A bi-drectional LSTM for hate-speech analysis
    """

    def __init__(self, embedding_matrix, embedding_dim, max_len, X_additional=None):
        inp1 = Input(shape=(max_len,))
        x = Embedding(embedding_matrix.shape[0], embedding_dim, weights=[embedding_matrix])(inp1)
        x = Bidirectional(LSTM(256, return_sequences=True))(x)
        x = BatchNormalization()(x)
        x = Bidirectional(LSTM(150))(x)
        x = Dense(128, activation="relu")(x)
        x = Dropout(0.4)(x)
        x = Dense(64, activation="relu")(x)
        x = Dense(3, activation='softmax', name='classifier')(x)
        model = Model(inputs=inp1, outputs=x)

        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
        self.model = model


class BiLSTM2():
    """
    bidirectional lstm for hate-speech analysis
    """
    def __init__(self, embedding_matrix, embedding_dim, max_len):
        inp1 = Input(shape=(max_len,))
        #
        x = Embedding(embedding_matrix.shape[0], embedding_dim , weights=[embedding_matrix])(inp1)
        x = SpatialDropout1D(0.3)(x)
        x = Bidirectional(LSTM(embedding_dim, dropout=0.3, recurrent_dropout=0.3))(x)
        x = Dense(embedding_dim, activation='relu')(x)
        x = Dropout(0.4)(x)
        x = Dense(embedding_dim, activation='relu')(x)
        x = Dropout(0.4)(x)
        x = Dense(3, activation='softmax', name='classifier')(x)
        model = Model(inputs=inp1, outputs=x)
        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
        self.model = model


class LSTM3():
    """
    LSTM for hate-speech analysis
    """
    def __init__(self, embedding_matrix, embedding_dim, max_len):
        inp1 = Input(shape=(max_len,))
        x = Embedding(embedding_matrix.shape[0], embedding_dim, input_length=max_len)(inp1)
        x = SpatialDropout1D(0.2)(x)
        x = LSTM(124, return_sequences=False)(x)
        x = Dropout(0.2)(x)
        x = Dense(3, activation='softmax', name='classifier')(x)
        model = Model(inputs=inp1, outputs=x)
        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])

        self.model = model


class LSTM4():
    def __init__(self, embedding_matrix, embedding_dim, max_len):
        main_input1 = Input(shape=(max_len,), name='main_input1')
        x1 = (Embedding(embedding_matrix.shape[0], embedding_dim, input_length=max_len,
                        weights=[embedding_matrix], trainable=False))(main_input1)
        x1 = SpatialDropout1D(0.4)(x1)
        x2 = Bidirectional(LSTM(75, dropout=0.5, return_sequences=True))(x1)
        x = Dropout(0.55)(x2)
        x = Bidirectional(LSTM(50, dropout=0.5, return_sequences=True))(x)
        hidden = concatenate([
            x1,
            x2,
            x
        ])
        hidden = Dense(32, activation='selu')(hidden)
        hidden = Dropout(0.5)(hidden)
        hidden = Dense(16, activation='selu')(hidden)
        hidden = Dropout(0.5)(hidden)
        output_lay1 = Dense(3, activation='sigmoid')(hidden)

        self.model = Model(inputs=[main_input1], outputs=output_lay1)

