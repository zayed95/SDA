import gensim.downloader as api
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from gensim.models import Word2Vec
from src.enums import RepresentationMethod
import pandas as pd
import numpy as np

class Represent:
    def __init__(self, column: pd.Series):
        self.col = column
        self.model = None
        
    def text_representation(self, method: RepresentationMethod) -> pd.DataFrame:

        if method == RepresentationMethod.BOW.value:
            self.model = CountVectorizer()
            feature_array = self.model.fit_transform(self.col)
            return pd.DataFrame(feature_array.toarray(), columns=self.model.get_feature_names_out())
        
        if method == RepresentationMethod.TF_IDF.value:
            self.model = TfidfVectorizer(max_features=5000)
            feature_array = self.model.fit_transform(self.col)
            return pd.DataFrame(feature_array.toarray(), columns=self.model.get_feature_names_out())


        if method == RepresentationMethod.WORD2VEC.value:
            sentences = self.col.apply(lambda x: x.split() if isinstance(x, str) else x)
            
            self.model = Word2Vec(
                sentences=sentences,
                vector_size=100,
                window=5,
                min_count=1
            )

            doc_vectors = []
            for doc in sentences:
                vecs = [self.model.wv[word] for word in doc if word in self.model.wv]
                if vecs:
                    doc_vectors.append(np.mean(vecs, axis=0))
                else:
                    doc_vectors.append(np.zeros(100))

            return pd.DataFrame(doc_vectors, columns=[f"v_{i}" for i in range(100)])

        
        if method == RepresentationMethod.GLOVE.value:

            self.model = api.load("glove-wiki-gigaword-100")
            sentences = self.col.apply(lambda x: x.split() if isinstance(x, str) else x)

            doc_vectors = []
            for doc in sentences:
                vecs = [self.model[w] for w in doc if w in self.model]
                if vecs:
                    doc_vectors.append(np.mean(vecs, axis=0))
                else:
                    doc_vectors.append(np.zeros(100))

            return pd.DataFrame(doc_vectors, columns=[f"v_{i}" for i in range(100)])

