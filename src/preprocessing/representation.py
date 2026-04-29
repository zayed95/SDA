from curses import window
import tokenize
import gensim.downloader as api
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from gensim.models import Word2Vec
from enums import RepresentationMethod
import pandas as pd

class represent:
    def __init__(self, method: RepresentationMethod, df: pd.DataFrame, column):
        self.method = method
        self.df = df
        self.col = column
        self.model = None
        
    def text_representation(self) -> pd.DataFrame:

        if self.method == RepresentationMethod.BOW.value:
            self.model = CountVectorizer()
            feature_array = self.model.fit_transform(self.df[self.col])
            return pd.DataFrame(feature_array.toarray(), columns=self.model.get_feature_names_out)
        
        if self.method == RepresentationMethod.WORD2VEC.value:
            tokenized_text = self.df[self.col].apply(lambda x: x.split())
            self.model = Word2Vec(
                sentences=tokenized_text,
                vector_size=100,
                window=5,
                min_count=3
            )

            vectors = [self.model.wv[word] for word in tokenized_text if word in self.model.wv]

            return pd.DataFrame(vectors.tolist(), columns=[f"v_{i}" for i in range(100)])
        
        if self.method == RepresentationMethod.TF_IDF.value:
            self.model = TfidfVectorizer(max_features=5000)
            feature_array = self.model.fit_transform(self.df[self.col])
            return pd.DataFrame(feature_array.toarray(), columns=self.model.get_feature_names_out)
        
        if self.method == RepresentationMethod.GLOVE.value:

            self.model = api.load("glove-wiki-gigaword-100")
            