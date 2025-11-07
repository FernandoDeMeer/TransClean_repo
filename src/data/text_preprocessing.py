from sklearn.feature_extraction.text import TfidfVectorizer
import string
import numpy as np
import re
from tqdm.auto import tqdm

import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)


def get_top_tf_idf_words(feature_names, response, top_n=2):

    sorted_nzs = np.argsort(response.data)[:-(top_n + 1):-1]
    return feature_names[response.indices[sorted_nzs]]


class TextProcessor:

    def __init__(self, language='english', stem=False, tf_idf=False):
        self.stem = stem
        self.language = language
        self.tf_idf = tf_idf

    def normalize_text(self, text: str):
        stop_words = set(stopwords.words(self.language))
        punct_words = set(string.punctuation)
        # Common uninformative terms in sentences describing company name changes in company descriptions
        punct_words.update(['company', 'formerly', 'known', 'changed', 'name'])
        normalized = [w for w in word_tokenize(text.lower())
                      if (w not in punct_words)
                      and (w not in stop_words)]

        if self.stem:
            stemmer = SnowballStemmer(self.language)
            normalized = [stemmer.stem(w) for w in normalized]

        return ' '.join(normalized)

    def tf_idf_ordering(self, df):
        if self.tf_idf:
            vectorizer = TfidfVectorizer()
            train_tf = vectorizer.fit(df['description'].fillna('').values)
            feature_array = np.array(train_tf.get_feature_names())

            def apply_tfidf(desc):
                top_tf_idf_words = get_top_tf_idf_words(feature_array, train_tf.transform(
                    [desc]), int(len(desc.split(' ')) * .5))
                low_tf_idf_words = set(feature_array[train_tf.transform([desc]).indices]) - set(top_tf_idf_words)
                for word in low_tf_idf_words:
                    desc = re.sub(r'(^|\s+)' + word + r'($|\s+)', ' ', desc)
                return desc

            df.loc[df['description'].apply(type) == str, 'description'] = df.loc[df['description'].apply(type) == str, 'description'] \
                .swifter \
                .allow_dask_on_strings(enable=True) \
                .progress_bar(desc='[Textprocessor] Applying TF-IDF on descriptions...') \
                .apply(apply_tfidf)

        return df

    def remove_punctuation_characters(self, txt: str):
        clean_str = re.sub(r"""
                       [,.;@#?!&$/()-]+  # Accept one or more copies of punctuation
                       \ *           # plus zero or more copies of a space,
                       """,
                           " ",  # and replace it with a single space
                           txt, flags=re.VERBOSE)
        return clean_str

