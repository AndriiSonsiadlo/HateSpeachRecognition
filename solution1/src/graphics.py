import random
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import seaborn as sb


def df_pie(df):
    plt.pie(df['class'].value_counts().values,
            labels=df['class'].value_counts().index,
            autopct='%1.1f%%')
    plt.show()


def random_color_func(word=None, font_size=None, position=None, orientation=None, font_path=None, random_state=None):
    h = 344
    s = int(100.0 * 255.0 / 255.0)
    l = int(100.0 * float(random_state.randint(60, 120)) / 255.0)
    return "hsl({}, {}%, {}%)".format(h, s, l)


def plot_wordcloud(df, cat_type):
    stop_words = set(STOPWORDS)
    stop_words.add("rt")

    wordcloud = WordCloud(
        background_color='white',
        stopwords=stop_words,
        max_words=200,
        max_font_size=60,
        random_state=42
    ).generate(str(df.loc[df["category"] == cat_type].text))
    print(wordcloud)
    fig = plt.figure(1)
    plt.imshow(wordcloud.recolor(color_func=random_color_func, random_state=3),
               interpolation="bilinear")
    plt.axis('off')
    plt.show()


def show_count_plot(data, col='label'):
    sb.countplot(col, data=data)