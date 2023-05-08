# custom imports
from definitions import ROOT_DIR
from embeddings import *
from text_processing import *
from graphics import *
from Model import *

# tools
import numpy as np
import pandas as pd
import os
from collections import Counter

# Text Pre-processing libraries
import warnings
import nltk
from nltk import word_tokenize, sent_tokenize, FreqDist
from nltk.stem import LancasterStemmer
from nltk.tokenize import TweetTokenizer

# TODO: uncomment on first run
# nltk.download('stopwords')
# nltk.download('omw-1.4')
# nltk.download('wordnet')
# nltk.download('punkt')
warnings.filterwarnings('ignore')

from imblearn.over_sampling import RandomOverSampler, SMOTE
from sklearn.feature_extraction.text import TfidfVectorizer
import preprocessor as p
from ekphrasis.classes.segmenter import Segmenter

# Tensorflow imports to build the model.
import tensorflow as tf
from tensorflow import keras
from keras import layers
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder


# CONSTANTS, settings
DATA_DIR = 'data'
EMBED_DIR = 'data/embeds'
DATA_FILE = 'labeled_data.csv'
EMBED_FILE = 'glove.twitter.27B.200d.txt'
MAX_WORDS = 5000
MAX_LEN = 100
desired_width = 320
pd.set_option('display.width', desired_width)
np.set_printoptions(linewidth=desired_width)
pd.set_option('display.max_columns', 10)


def split_df(df, feat='clean_text', tar='label'):
    features = df[feat]
    target = df[tar]

    x_train, x_test, y_train, y_test = train_test_split(features,
                                                        target,
                                                        test_size=0.2,
                                                        random_state=22)

    print(f" x_train shape: {x_train.shape}, x_val shape: {x_test.shape}")

    return x_train, x_test, y_train, y_test


def balance_df(x_train, y_train):
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(x_train, y_train)
    return X_res, y_res


def padding_data(data, max_len=MAX_LEN):
    tokenizer = Tokenizer(num_words=MAX_WORDS,
                          lower=True,
                          split=' ')
    tokenizer.fit_on_texts(data)
    sequences = tokenizer.texts_to_sequences(data)
    x = pad_sequences(sequences, maxlen=max_len, padding='post', truncating='post')
    return pd.DataFrame(x)


def create_model():
    model = keras.models.Sequential([
        layers.Embedding(MAX_WORDS, 32, input_length=MAX_LEN),
        layers.Bidirectional(layers.LSTM(16)),
        layers.Dense(512, activation='relu', kernel_regularizer='l1'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(3, activation='softmax')
    ])

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    print("model summary", model.summary())

    return model


def bag_words(x_train, x_test):
    tfidf_vectorizer = TfidfVectorizer(use_idf=True)

    X_train_vectors_tfidf = tfidf_vectorizer.fit_transform(x_train)
    X_test_vectors_tfidf = tfidf_vectorizer.transform(x_test)

    return X_train_vectors_tfidf, X_test_vectors_tfidf


def word_2_vec(x_train, x_test):
    X_train_tok = [nltk.word_tokenize(i) for i in x_train]
    X_test_tok = [nltk.word_tokenize(i) for i in x_test]

    return X_train_tok, X_test_tok


def train(model, xtrain, ytrain, xtest, ytest, epochs=100):
    es = EarlyStopping(patience=3,
                       monitor='val_accuracy',
                       restore_best_weights=True)

    lr = ReduceLROnPlateau(patience=2,
                           monitor='val_loss',
                           factor=0.5,
                           verbose=0)

    history = model.fit(xtrain, ytrain,
                        validation_data=(xtest, ytest),
                        epochs=epochs,
                        verbose=1,
                        batch_size=32,
                        callbacks=[lr, es])

    return model, history


def df_cleanup(df0):
    c = df0['class']
    df0.rename(columns={'tweet': 'text',
                        'class': 'category'},
               inplace=True)
    a = df0['text']
    b = df0['category'].map({0: 'hate_speech', 1: 'offensive_language', 2: 'neither'})

    df = pd.concat([a, b, c], axis=1)
    df.rename(columns={'class': 'label'},
              inplace=True)

    return df


def print_dataset_stats(df):
    hate, offensive, neither = np.bincount(df['label'])
    total = hate + offensive + neither
    print('Examples:\n    Total: {}\n    hate: {} ({:.2f}% of total)\n'.format(
        total, hate, 100 * hate / total))
    print('Examples:\n    Total: {}\n    Ofensive: {} ({:.2f}% of total)\n'.format(
        total, offensive, 100 * offensive / total))
    print('Examples:\n    Total: {}\n    Neither: {} ({:.2f}% of total)\n'.format(
        total, neither, 100 * neither / total))


def calculateMetrics(ypred, ytrue):
    acc = accuracy_score(ytrue, ypred)
    f1 = f1_score(ytrue, ypred)
    f1_average = f1_score(ytrue, ypred, average="macro")
    return " f1 score: " + str(round(f1, 3)) + " f1 average: " + str(round(f1_average, 3)) + " accuracy: " + str(
        round(acc, 3))


def one_hot_encode_labels(data):
    y = data.values
    ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
    y = np.array(ct.fit_transform(y))

    return y


def run():
    file_path = os.path.join(ROOT_DIR, DATA_DIR, DATA_FILE)
    df0 = pd.read_csv(file_path)
    df = df_cleanup(df0)
    print(df.columns.values)
    # cleans tweets -- URLs, Mentions, etc
    # for i, v in enumerate(df['text']):
    #     df.loc[i, 'text'] = p.clean(v)
    # df_pie(train_df)
    # show_count_plot("label", df)

    df['clean_text'] = df['text'].apply(lambda text: preprocess(text))

    # padding sequence -


    x_train, x_test, y_train, y_test = split_df(df)

    # one-hot encode labels
    y_train = one_hot_encode_labels(pd.DataFrame(y_train))
    y_test = one_hot_encode_labels(pd.DataFrame(y_test))
    print(y_train.shape, y_test.shape)

    embed_path = os.path.join(ROOT_DIR, EMBED_DIR, EMBED_FILE)
    embed_dim = 200

    Xtrain = x_train.tolist()
    Ytrain = y_train.tolist()
    Xtest = x_test.tolist()
    Ytest = y_test.tolist()

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(Xtrain)

    # Creating the embedding matrix
    embedding = Embeddings(embed_path, embed_dim)
    embedding_matrix = embedding.create_embedding_matrix(tokenizer, len(tokenizer.word_counts))

    # Creating the padded input for the deep learning model
    max_len = np.max([len(text.split()) for text in Xtrain])
    t2t = TextToTensor(
        tokenizer=tokenizer,
        max_len=max_len
    )

    Xtrain = t2t.string_to_tensor(Xtrain)

    lstm = LSTMmodel(embedding_matrix=embedding_matrix,
                     embedding_dim=embed_dim,
                     max_len=max_len)

    lstm.model.fit(np.array(Xtrain), np.array(Ytrain), batch_size=128, epochs=50)

    model = lstm.model

    # If X_test is provided we make predictions with the created model

    Xtest = t2t.string_to_tensor(Xtest)
    yhat = [x[0] for x in lstm.model.predict(Xtest).tolist()]

    acc = 0
    # If true labels are provided we calculate the accuracy of the model
    if len(Ytest) > 0:
        acc = accuracy_score(Ytest, [1 if x > 0.5 else 0 for x in yhat])

    print(f"acc= {round(acc, 2)}")




# -------------- OLD CODE ------------------------
    # Getting longest sentence
    # max_len = np.max([len(text.split()) for text in x_train])
    # print(f"max len words: {max_len}")
    #
    #
    #
    # # Tokenize some shit
    # tokenizer = Tokenizer()
    # tokenizer.fit_on_texts(x_train)
    #
    # # Create embedding matrix using twitter embed file:
    # embedding = Embeddings(embed_path, embed_dim)
    # embedding_matrix = embedding.create_embedding_matrix(tokenizer, len(tokenizer.word_counts))




def run_smote():
    file_path = os.path.join(ROOT_DIR, DATA_DIR, DATA_FILE)
    df0 = pd.read_csv(file_path)
    df = df_cleanup(df0)

    df['clean_text'] = df['text'].apply(lambda text: preprocess(text))

    #TODO: sequence padding before splitting so that strings of same length

    # splitting train and test
    x_train, x_test, y_train, y_test = split_df(df)

    y_train = one_hot_encode_labels(pd.DataFrame(y_train))
    y_test = one_hot_encode_labels(pd.DataFrame(y_test))
    print(y_train.shape, y_test.shape)

    # TODO: implement imbalanced data alg (e.g. SMOTE)
    print(x_train.shape, y_train.shape)

    # vectorizing using bog-of-words
    x_train_b, x_test_b = bag_words(x_train, x_test)

    # generating synthetic data with smote
    x_train_smote, y_train_smote = balance_df(x_train_b, y_train)

    print(x_train_smote.shape, y_train_smote.shape)

    # Vectorize text before fitting model
    # Word2Vec
    # x_tr, x_tst = word_2_vec(x_train, x_test)
    #
    # model = create_model()
    # model, hist = train(model, xtrain=x_tr, xtest=x_tst, ytrain=y_train, ytest=y_test)
    # y_preds = model.predict(x_test)
    # print(f"accuracy: {accuracy_score(y_test, y_preds)}")


def run_oversample():


    # ---------- balance dataset using Oversampling --------------
    # vectorizer = TfidfVectorizer()
    # vectorizer.fit(x_train)
    # x_train_tf = vectorizer.transform(x_train)
    #
    # x_test_tf = vectorizer.transform(x_test)
    # x_test_tf = x_test_tf.toarray()
    #
    # ROS = RandomOverSampler(sampling_strategy=1)
    # x_train_ros, y_train_ros = ROS.fit_resample(x_train_tf, y_train)
    # -----------------------------------------------------------------------
    pass


if __name__ == '__main__':
    # run_smote()
    # run_oversample()
    run()
