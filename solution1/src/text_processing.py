import re, string, unicodedata

from nltk import TweetTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.probability import FreqDist

# TODO: slang word removal

def remove_numbers(text):
    number_pattern = r'\d+'
    without_number = re.sub(pattern=number_pattern, repl=" ", string=text)
    return without_number


def remove_punctuations(text):
    # Removing punctuations present in the text
    punctuations_list = string.punctuation

    temp = str.maketrans('', '', punctuations_list)
    return text.translate(temp)


def remove_stopwords(text):
    stop_words = set(stopwords.words('english'))
    stop_words.add('rt')
    # stop_words.add('RT')
    imp_words = []

    # Storing the important words
    for word in str(text).split():

        if word not in stop_words:
            lemmatizer = WordNetLemmatizer()
            lemmatizer.lemmatize(word)
            imp_words.append(word)

    output = " ".join(imp_words)

    return output


def remove_extra_white_spaces(text):
    single_char_pattern = r'\s+[a-zA-Z]\s+'
    without_sc = re.sub(pattern=single_char_pattern, repl=" ", string=text)
    return without_sc


def lemmetize(text, lang='english'):
    lemmatizer = WordNetLemmatizer()
    w_tokenizer = TweetTokenizer()
    # tokens = word_tokenize(text, language=lang)
    # for i in range(len(tokens)):
    #     lemma_word = lemmatizer.lemmatize(tokens[i])
    #     tokens[i] = lemma_word
    # return " ".join(tokens)
    tokens = [(lemmatizer.lemmatize(w)) for w in w_tokenizer.tokenize(text)]
    return " ".join(tokens)


def preprocess(text):
    # reduce tweets to lower_case
    text = text.lower()

    # remove links
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"www.\S+", "", text)

    # remove hashtags
    text = re.sub("@[A-Za-z0-9_]+", "", text)
    text = re.sub("#[A-Za-z0-9_]+", "", text)

    text = remove_punctuations(text)
    text = remove_numbers(text)
    text = remove_extra_white_spaces(text)
    text = remove_stopwords(text)
    text = text.strip()
    text = lemmetize(text)

    # text = remove_extra_white_spaces(text)

    return text


class TextToTensor():

    def __init__(self, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.max_len = max_len

    def string_to_tensor(self, string_list: list) -> list:
        """
        A method to convert a string list to a tensor for a deep learning model
        """
        string_seq = self.tokenizer.texts_to_sequences(string_list)
        padded_seq = pad_sequences(string_seq, maxlen=self.max_len)

        return padded_seq

    def tweets_to_tensor(self, tweet_list: list) -> list:

        if self.tokenizer.isInstance(TweetTokenizer):
            tokenized = TweetTokenizer(self.tokenizer).tokenize_sents(tweet_list)
            # Flatten the tokenized_tweets list
            flat_tokens = [token for sublist in tokenized for token in sublist]

            # Build vocabulary and assign indices
            fdist = FreqDist(flat_tokens)
            vocabulary = {word: index + 1 for index, (word, _) in enumerate(fdist.most_common())}

            # Encode the sequences
            encoded_sequences = [[vocabulary[token] for token in tokens] for tokens in tokenized]

            print(encoded_sequences[:3])


