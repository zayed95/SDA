import time, os, argparse, tqdm, logging, emoji, re
import pandas as pd
import nltk
import pkg_resources
from symspellpy import SymSpell
from langdetect import detect, DetectorFactory
from deep_translator import GoogleTranslator
from textblob import TextBlob, Word
from nltk.corpus import stopwords

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


nltk.download('stopwords', quiet=True)
STOPWORDS = set(stopwords.words('english'))

DetectorFactory.seed = 0

class Preprocessing:

    def __init__(self) -> None:
        self.sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
        self.dictionary_path = pkg_resources.resource_filename(
        "symspellpy", "frequency_dictionary_en_82_765.txt"
    )

    # Function to drop duplicates
    def clean_df(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.drop_duplicates()

    # A function to detect the language    
    def detect_language(self, text: str) -> str:
        try:
            return detect(text)
        except:
            logger.info("Unknown language found")
            return 'unknown'
    
    # A function to translate the tweet or drop the record altogether
    def handle_language(self, df: pd.DataFrame, mode='drop') -> pd.DataFrame:

        df['language'] = df['text'].apply(self.detect_language)
        df = df[df['language'] != 'unknown'].reset_index(drop=True)
        df['text'].apply(self.translate)

        return df.drop(columns=['language'])

    # A function to translate to English if the text was in a different language
    def translate(self, text: str) -> str:

        if self.detect_language(text) == 'en':
            return text
        
        try:
            return GoogleTranslator(source='auto', target='en')
        except:
            logger.error("Failed to trnaslate to English")
            return text
        
    # Replaces emojis
    def remove_emoji(self, text: str) -> str:
        return emoji.demojize(text)
    
    def remove_urls(self, text: str) -> str: 
        return re.sub(r'https?://\S+|www\.\S+', '', str(text))
    
    def fix_spelling(self, text: str) -> str:
        suggestions = self.sym_spell.lookup_compound(text, max_edit_distance=2)
        return suggestions[0].term
    
    def remove_punctuation(self, text: str) -> str:
        return re.sub(r'[^\w\s]', '', str(text)).lower() 
    
    def tokenize(self, text: str) -> str:
        return " ".join(TextBlob(text).words)
    
    def stop_words_removal(self, text: str) -> str:
        return " ".join([w for w in text.split() if w not in STOPWORDS])
    
    def lemmetize(self, text: str) -> str:
        blob = TextBlob(text)
        filtered = [word for word, tag in blob.tags if tag.startswith('NN') or tag.startswith('VB')]
        return " ".join([Word(w).lemmatize() for w in filtered])