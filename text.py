import re
import nltk
import numpy as np

from typing import Set
from wordcloud import WordCloud
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)


def get_stop_words() -> Set[str]:
    sws = set(stopwords.words('english'))
    custom_words = ['new york times', 'mrs', 'mr', 'said',
                    'one', 'way',
                    'york', 'little', 'new', 'even', 'im']
    return set([w.lower() for w in sws.union(custom_words)])


stop_words = get_stop_words()


def create_wordcloud(texts: np.ndarray) -> WordCloud:
    words = build_text(texts)
    wordcloud = WordCloud(
        background_color='white',
        max_words=200,
        max_font_size=40,
        scale=3,
        random_state=1
    ).generate(words)
    return wordcloud


def build_text(texts: np.ndarray) -> str:
    joined_text = ' '.join(texts)
    filtered_text = filter_stop_words(joined_text)
    return filtered_text


def filter_stop_words(text: str) -> str:
    filtered = text.lower()
    filtered = re.sub(r'[^\w\s]', '', filtered)
    return ' '.join(
        [w for w in word_tokenize(filtered) if not w in stop_words]
    )


def is_stop_word(word: str) -> bool:
    return (word in stop_words)
