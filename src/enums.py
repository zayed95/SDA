from enum import Enum

class RepresentationMethod(Enum):

    BOW = "bag_of_words"
    GLOVE = "glove"
    TF_IDF = "tf_idf"
    WORD2VEC = "word2vec"