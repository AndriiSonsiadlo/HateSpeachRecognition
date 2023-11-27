# custom imports
import argparse

import matplotlib.pyplot as plt

# from definitions import ROOT_DIR
ROOT_DIR = '../../'
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
# Torch imports

# Tensorflow imports to build the model.
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras import backend as K
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# other
from gensim.models import Word2Vec
from imblearn.under_sampling import RandomUnderSampler
from transformers import TFAutoModel, AutoTokenizer

# CONSTANTS, settings
DATA_DIR = 'data'
OUTPUT_DIR = 'data/output'
EMBED_DIR = 'data/embeds'
DATA_FILE = 'labeled_data.csv'
EMBED_FILE = 'glove.twitter.27B.200d.txt'
MAX_WORDS = 2000
MAX_LEN = 300
desired_width = 320
pd.set_option('display.width', desired_width)
np.set_printoptions(linewidth=desired_width)
pd.set_option('display.max_columns', 10)


def split_df(df, feat='clean_text', tar='label'):
    """
    splits text data from DataFrame into train and test
    :param df:
    :param feat: column name for features
    :param tar: column name for targets
    :return: train and test split
    """
    features = df[feat]
    target = df[tar]

    x_train, x_test, y_train, y_test = train_test_split(features,
                                                        target,
                                                        test_size=0.2,
                                                        random_state=22)

    print(f" x_train shape: {x_train.shape}, x_val shape: {x_test.shape}")

    return x_train, x_test, y_train, y_test


def smote_balance_train(x_train, y_train, seed_n=42):
    """
    balances data using SMOTE
    :param seed_n: seed number
    :param x_train: x_train vectorized data
    :param y_train: y_train vectorized data
    :return:
    """
    sm = SMOTE(random_state=seed_n)
    X_res, y_res = sm.fit_resample(x_train, y_train)
    return X_res, y_res


def padding_data(data, max_len=MAX_LEN):
    """
    padding text strings
    :param data:
    :param max_len:
    :return:
    """
    tokenizer = Tokenizer(num_words=MAX_WORDS,
                          lower=True,
                          split=' ')
    tokenizer.fit_on_texts(data)
    sequences = tokenizer.texts_to_sequences(data)
    x = pad_sequences(sequences, maxlen=max_len, padding='post', truncating='post')
    return pd.DataFrame(x)


def df_cleanup(df0):
    """
    cleans raw dataframe creating new columns for text data preprocessing
    :param df0:
    :return: cleaned DataFrame
    """
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
    """
    prints DataFrame stats for hate, offensive, and neither text
    :param df:
    :return:
    """
    hate, offensive, neither = np.bincount(df['label'])
    total = hate + offensive + neither
    print('Examples:\n    Total: {}\n    hate: {} ({:.2f}% of total)\n'.format(
        total, hate, 100 * hate / total))
    print('Examples:\n    Total: {}\n    Offensive: {} ({:.2f}% of total)\n'.format(
        total, offensive, 100 * offensive / total))
    print('Examples:\n    Total: {}\n    Neither: {} ({:.2f}% of total)\n'.format(
        total, neither, 100 * neither / total))


def calculateMetrics(y_true, y_pred):
    """
    calculates precision, f1 score, and recall values
    :param y_true: label targets
    :param y_pred: label predictions
    :return:
    """
    # precision
    acc = accuracy_score(y_true, y_pred)
    # f1 score
    f1 = f1_score(y_true, y_pred)
    f1_average = f1_score(y_true, y_pred, average="macro")
    # recall
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return f" f1 score: {str(round(f1, 3))} f1 average: {str(round(f1_average, 3))} accuracy: {str(round(acc, 3))} recall: {str(round(recall, 3))}"


def one_hot_encode_labels(data):
    """
    converts
    :param data:
    :return:
    """
    y = data.values
    ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
    y = np.array(ct.fit_transform(y))

    return y


def plot_bargraph_by_category(df, category="category", filename=None):
    """

    :param df: pandas DateFrane
    :param category: x value by which to group
    :return:
    """
    sb.countplot(data=df, x=category)
    if filename is None:
        plt.show()
    else:
        plt.savefig(filename, bbox_inches='tight')
    plt.clf()


def plot_hist(history, category: str, filename=None):
    """
    summarize history for loss

    :param filename: where to save figure
    :param category: loss or accuracy
    :param history: history from model.fit()
    :return:
    """
    #

    plt.plot(history.history[category])
    plt.plot(history.history['val_' + category])
    plt.xlabel("Epochs")
    plt.title(f"Model {category.capitalize()}")
    plt.ylabel(category.capitalize())
    plt.legend(['train', 'test'], loc='upper left')

    if filename is None:
        plt.show()
    else:
        plt.savefig(filename, bbox_inches='tight')
    plt.clf()


def word2vec_embed_matrix(sentences, v_dim, mode="bow"):
    """
    A method to create the embedding matrix using Word2Vec
    """
    tokenizer = TweetTokenizer()
    tokenized_tweets = tokenizer.tokenize_sents(sentences)

    if mode == "bow":
        sg = 0
    else:
        sg = 1
    model = Word2Vec(tokenized_tweets, vector_size=v_dim, window=5, min_count=1, workers=4, sg=sg)
    vocab = list(model.wv.index_to_key)
    print(f"vocab size: {len(vocab)}")
    max_features = len(vocab)
    embedding_dim = model.vector_size
    print(f"embedding_dim={embedding_dim}")
    embedding_matrix = np.zeros((max_features + 1, embedding_dim))
    for i, word in enumerate(vocab):
        if i > max_features:
            break
        else:
            try:
                embedding_vector = model.wv[word]
                if embedding_vector is not None:
                    embedding_matrix[i] = embedding_vector
            except:
                continue

    return embedding_matrix


def glove_embed_matrix(tokenized, vocab_size, tokenizer, embed_dim=200):
    """
    creates an embed matrix using stanford glove file
    :param embed_dim:
    :param tokenizer:
    :return:
    """
    kris_path = r"C:\Users\Krzysztof\PycharmProjects\HateSpeachRecognition\embeds\glove.twitter.27B.200d.txt"
    embed_path = os.path.join(ROOT_DIR, EMBED_DIR, EMBED_FILE)

    # Creating the embedding matrix using stanford GloVe
    embedding = Embeddings(embed_path, embed_dim)
    return embedding.create_embedding_matrix(max_features=vocab_size, tokenized_tweets=tokenized, tokenizer=tokenizer)


def bert_embed_matrix(text_dataset):
    # Load the BERT model and tokenizer
    model_name = 'bert-base-uncased'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = TFAutoModel.from_pretrained(model_name)

    tokenized_dataset = tokenizer(text_dataset, padding=True, truncation=True, return_tensors='tf')
    # Obtain BERT embeddings
    embeddings = model(**tokenized_dataset)[0]
    return embeddings.numpy()


def balance_train(x_train, y_train, mode="smote"):
    if mode == "smote":
        return smote_balance_train(x_train, y_train)
    elif mode == "us":
        undersampler = RandomUnderSampler(sampling_strategy='majority')
        return undersampler.fit_resample(x_train, y_train)
    else:
        # no change
        return np.array(x_train), np.array(y_train)


def softmax_to_one_hot(softmax):
    tensor = tf.one_hot(tf.argmax(softmax, axis=1), depth=3)
    return tensor.numpy()


def save_results(lstm, xtest, ytest, history, save_file):
    fn = save_file.split('.')[0]
    dir_path = f"../../data/output/{fn}"
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        print(f"Directory '{dir_path}' created.")
    else:
        print(f"Directory '{dir_path}' already exists.")

    plot_hist(history, category="accuracy", filename=os.path.join(dir_path, f"{fn}_accuracy.png"))
    plot_hist(history, category="loss", filename=os.path.join(dir_path, f"{fn}_loss.png"))

    # save model
    models_dir = os.path.join(ROOT_DIR, "solution1", "models")
    lstm.model.save(os.path.join(models_dir, save_file))

    preds = lstm.model.predict(xtest)

    preds = softmax_to_one_hot(preds)

    print(f"preds shape {preds.shape}, test shape {ytest.shape}")

    acc = accuracy_score(ytest, preds)

    print(f"acc= {round(acc, 2)}")

    report = classification_report(ytest, preds)

    # Save the report to a file
    output_file = f'{dir_path}/{fn}_classification_report.txt'
    with open(output_file, 'w') as file:
        file.write(report)


def save_stats_df(df):
    import dataframe_image as dfi
    # EDIT: see deprecation warnings below

    hate, offensive, neither = np.bincount(df['label'])
    total = hate + offensive + neither
    phate = round(100 * hate / total, 2)
    poff = round(100 * offensive / total, 2)
    pneither = round(100 * neither / total, 2)
    datas = [['hate', hate, phate], ['offensive', offensive, poff], ['neither', neither, pneither]]
    stats_df = pd.DataFrame(datas, columns=['Type of Speech', 'Samples', '% of Total'])
    dfi.export(stats_df, r'C:\Users\Krzysztof\Desktop\nlp\data_stats.png')


def tokenize_tweets(tweets_list):
    tokenizer = TweetTokenizer()
    return tokenizer.tokenize_sents(tweets_list)


def create_vocabulary(tokenized_tweets):
    # Flatten the tokenized_tweets list
    flat_tokens = [token for sublist in tokenized_tweets for token in sublist]

    # Build vocabulary and assign indices
    fdist = FreqDist(flat_tokens)
    vocabulary = {word: index + 1 for index, (word, _) in enumerate(fdist.most_common())}

    return vocabulary


def tweets_to_tensor(tweet_list: list, tokenizer: TweetTokenizer, max_len: int):
    if isinstance(tokenizer, TweetTokenizer):
        tokenized = tokenize_tweets(tweet_list)
        vocabulary = create_vocabulary(tokenized)

        # Encode the tokens
        encoded_sequences = [[vocabulary[token] for token in tokens] for tokens in tokenized]
        tensors = pad_sequences(encoded_sequences, maxlen=max_len, padding='pre', truncating='pre', dtype='int32',
                                value=0)

        return tensors, len(vocabulary)


def test_lstm1(xtrain, ytrain, xtest, ytest, embedding_matrix, embed_dim, max_len, num_epochs=100,
               save_file="bi_lstm1.h5"):
    lstm = LSTMmodel(embedding_matrix=embedding_matrix,
                     embedding_dim=embed_dim,
                     max_len=max_len)

    es = EarlyStopping(patience=5,
                       monitor='val_loss',
                       restore_best_weights=True)

    lr = ReduceLROnPlateau(patience=5,
                           monitor='val_loss',
                           factor=0.5,
                           verbose=0)

    print("fitting model...")
    history = lstm.model.fit(xtrain, ytrain,
                             validation_split=0.33,
                             validation_data=(xtest, ytest),
                             batch_size=256,
                             epochs=num_epochs,
                             verbose=1,
                             callbacks=[es]
                             )

    print(f"max_len={max_len}")
    print(f"embed_dim={embed_dim}")
    print(f"embed_matrix shape0={embedding_matrix.shape[0]}")

    save_results(lstm, xtest, ytest, history=history, save_file=save_file)


def test_lstm2(xtrain, ytrain, xtest, ytest, embedding_matrix, embed_dim, max_len, num_epochs=100,
               save_file="bi_lstm2.h5"):
    lstm = BiLSTM2(embedding_matrix=embedding_matrix, embedding_dim=embed_dim, max_len=max_len)
    #
    # early_stop = EarlyStopping(monitor='val_loss', patience=5)
    # history = lstm.model.fit(xtrain,
    #                          ytrain,
    #                          epochs=num_epochs,
    #                          validation_data=(xtest, ytest),
    #                          callbacks=[early_stop],
    #                          verbose=1)
    es = EarlyStopping(patience=5,
                       monitor='val_loss',
                       restore_best_weights=True)

    lr = ReduceLROnPlateau(patience=5,
                           monitor='loss',
                           factor=0.5,
                           verbose=0)

    print("fitting model...")
    history = lstm.model.fit(xtrain, ytrain,
                             validation_split=0.33,
                             validation_data=(xtest, ytest),
                             batch_size=128,
                             epochs=num_epochs,
                             verbose=1,
                             callbacks=[es]
                             )

    print(f"max_len={max_len}")
    print(f"embed_dim={embed_dim}")
    print(f"embed_matrix shape0={embedding_matrix.shape[0]}")

    save_results(lstm, xtest, ytest, history=history, save_file=save_file)


def test_lstm3(xtrain, ytrain, xtest, ytest, embedding_matrix, embed_dim, max_len, num_epochs=100,
               save_file="lstm3.h5"):
    lstm = LSTM3(embedding_matrix=embedding_matrix, embedding_dim=embed_dim, max_len=max_len)
    early_stop = EarlyStopping(monitor='val_loss', patience=5)
    history = lstm.model.fit(xtrain,
                             ytrain,
                             epochs=num_epochs,
                             validation_data=(xtest, ytest),
                             callbacks=[early_stop],
                             verbose=1)

    save_results(lstm, xtest, ytest, history=history, save_file=save_file)


def run():
    file_path = os.path.join(ROOT_DIR, DATA_DIR, DATA_FILE)

    parser = argparse.ArgumentParser(
        description='Tensorflow classificator for hate speech')
    parser.add_argument('--datapath', type=str, default=file_path,
                        help='path of dataset (default: data/labeled_data.csv)')
    parser.add_argument('--embeds', type=str, default="w2v-bow",
                        help='sets what embeddings to use (w2v-bow, w2v-sg, bert, or glove')
    parser.add_argument('--filename', type=str, default="lstm.h5",
                        help='model save filename')
    args = parser.parse_args()
    df0 = pd.read_csv(args.datapath)
    df = df_cleanup(df0)
    # print(df.columns.values)
    # cleans tweets -- URLs, Mentions, etc
    # for i, v in enumerate(df['text']):
    #     df.loc[i, 'text'] = p.clean(v)
    # df_pie(train_df)
    # show_count_plot("label", df)

    df['clean_text'] = df['text'].apply(lambda text: preprocess(text))

    # plot_bargraph_by_category(df, filename=r"C:\Users\Krzysztof\Desktop\nlp\data_plot.png")
    # save_stats_df(df)
    # print_dataset_stats(df)

    print(f"entries: {df.shape[0]}")

    x_train, x_test, y_train, y_test = split_df(df)

    # one-hot encode labels
    y_train = one_hot_encode_labels(pd.DataFrame(y_train))
    y_test = one_hot_encode_labels(pd.DataFrame(y_test))
    print(y_train.shape, y_test.shape)

    embed_dim = 200

    Xtrain = x_train.tolist()
    Ytrain = y_train.tolist()
    Xtest = x_test.tolist()
    Ytest = y_test
    max_len = np.max([len(text.split()) for text in Xtrain])

    tokenizer = TweetTokenizer()
    print("making int sequences")
    tokenized_train_tweets = tokenize_tweets(Xtrain)
    x_train_tensors, vocab_size = tweets_to_tensor(Xtrain, tokenizer=tokenizer, max_len=max_len)

    base_tokenizer = Tokenizer()
    base_tokenizer.fit_on_texts(Xtrain)

    # glove embeds
    print("making embed matrix..")
    if args.embeds == "glove":
        embedding_matrix = glove_embed_matrix(tokenized=tokenized_train_tweets, vocab_size=vocab_size, embed_dim=embed_dim, tokenizer=tokenizer)
    elif "w2v" in args.embeds:
        if "bow" in args.embeds.split('-'):
            # Word2Vec embeds
            embedding_matrix = word2vec_embed_matrix(Xtrain, embed_dim, mode="bow")
        else:
            embedding_matrix = word2vec_embed_matrix(Xtrain, embed_dim, mode="skip")
    else:
        # bert embeds
        embedding_matrix = bert_embed_matrix(Xtrain)

    # t2t = TextToTensor(
    #     tokenizer=tokenizer,
    #     max_len=max_len
    # )

    print(f"embed_m_sh: {embedding_matrix.shape} embed_dim: {embed_dim} max_len: {max_len}")

    Xtrain = x_train_tensors

    Xtest, _ = tweets_to_tensor(Xtest, tokenizer=tokenizer, max_len=max_len)
    Xtest = np.array(Xtest)

    print(np.array(Xtrain).shape)
    print(np.array(Ytrain).shape)

    x_train_balanced, y_train_balanced = balance_train(np.array(Xtrain), np.array(Ytrain), mode="us")

    print(f"x_balanced shape: {x_train_balanced.shape}")
    print(x_train_balanced[:5])
    print(f"y_balanced shape: {y_train_balanced.shape}")
    print(f"embed MATRIX: {embedding_matrix}")

    epochs = 100
    test_name = "us_sg_"

    test_lstm1(x_train_balanced, y_train_balanced, Xtest, Ytest,
               embedding_matrix=embedding_matrix,
               embed_dim=embed_dim,
               max_len=max_len,
               num_epochs=epochs,
               save_file=f"{test_name}bilstm1.h5")

    # test_lstm2(x_train_balanced, y_train_balanced, Xtest, Ytest,
    #            embedding_matrix=embedding_matrix,
    #            embed_dim=embed_dim,
    #            max_len=max_len,
    #            num_epochs=epochs,
    #            save_file="no_balance_es_val_loss_bilstm2.h5")

    test_lstm3(x_train_balanced, y_train_balanced, Xtest, Ytest,
               embedding_matrix=embedding_matrix,
               embed_dim=embed_dim,
               max_len=max_len,
               num_epochs=100,
               save_file=f"{test_name}lstm3.h5")


if __name__ == '__main__':
    # run_smote()
    # run_oversample()
    run()
