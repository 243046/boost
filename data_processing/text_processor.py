from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS as sk_words
import spacy
from nltk.corpus import stopwords


class TextProcessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.spc = spacy.load('en_core_web_sm')

    def _clean_with_spacy(self, x):
        return ' '.join([token.lemma_ for token in self.spc(x)])

    def _delete_stopwords(self, x):
        sw = list(set(stopwords.words('english') + list(sk_words)))
        return ' '.join([word for word in x.split() if word not in sw])

    def _clean_text(self, X):
        X = (
            X
            .str.lower()
            .str.replace(r'<.*?>', '', regex=True)  # html tags
            .str.replace(r'http\S+', '', regex=True)  # links
            .str.replace(r'[^\w\s\d]', '', regex=True)  # punctuation
            .str.replace(r'[\(\)]', '', regex=True)  # parentheses
            .str.replace(r'(\xa0)|(x000)|:|', '', regex=True)  # tilde, :
            .str.replace(r'_', ' ', regex=True)  # _
            .map(self._clean_with_spacy) # spacy lemmatizer
            .map(self._delete_stopwords) # stop words
            .str.replace(r'\s+', ' ', regex=True)  # whitespaces
        )
        return X

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return self._clean_text(X)
