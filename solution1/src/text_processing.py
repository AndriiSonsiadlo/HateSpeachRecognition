import re, string, unicodedata

from nltk import TweetTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


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
    stop_words.add('RT')
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

    # remove 'rt'
    text = re.sub("rt", "", text)

    text = remove_punctuations(text)
    text = remove_numbers(text)
    text = remove_extra_white_spaces(text)
    text = remove_stopwords(text)
    text = lemmetize(text)
    # text = remove_extra_white_spaces(text)

    return text