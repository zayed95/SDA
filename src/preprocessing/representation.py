from curses import window
import tokenize
import gensim.downloader as api
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from gensim.models import Word2Vec
from enums import RepresentationMethod
import pandas as pd
import numpy as np

class represent:
    def __init__(self, df: pd.DataFrame, column):
        self.df = df
        self.col = column
        self.model = None
        
    def text_representation(self, method: RepresentationMethod) -> pd.DataFrame:

        if method == RepresentationMethod.BOW.value:
            self.model = CountVectorizer()
            feature_array = self.model.fit_transform(self.df[self.col])
            return pd.DataFrame(feature_array.toarray(), columns=self.model.get_feature_names_out)
        
        if method == RepresentationMethod.TF_IDF.value:
            self.model = TfidfVectorizer(max_features=5000)
            feature_array = self.model.fit_transform(self.df[self.col])
            return pd.DataFrame(feature_array.toarray(), columns=self.model.get_feature_names_out)


        if method == RepresentationMethod.WORD2VEC.value:
            
            self.model = Word2Vec(
                sentences=self.df[self.col],
                vector_size=100,
                window=5,
                min_count=3
            )

            vectors = [self.model.wv[word] for word in tokens if word in self.model.wv]

            if not vectors:
                vectors = np.zeros(100)
            
            vectors = np.mean(vectors, axis=0)

            return pd.DataFrame(vectors.tolist(), columns=[f"v_{i}" for i in range(100)])

        
        if method == RepresentationMethod.GLOVE.value:

            self.model = api.load("glove-wiki-gigaword-100")

            vectors = [self.model[w] for w in self.df[self.col] if w in self.model]

            if not vectors:
                vectors = np.zeros(100)

            vectors = np.mean(vectors, axis=0)

            return pd.DataFrame(vectors.tolist(), columns=[f"v_{i}" for i in range(100)])

