import re

import emoji
import joblib
import nltk
import pandas as pd
import requests
from camel_tools.utils.charsets import AR_LETTERS_CHARSET
from camel_tools.utils.dediac import dediac_ar
from nltk.stem.isri import ISRIStemmer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer

STOPWORDS_URL = "https://raw.githubusercontent.com/mohataher/arabic-stop-words/master/list.txt"

class TextProcessingPipeline:
    def __init__(self):
        nltk.download('punkt', raise_on_error=True)
        nltk.download('punkt_tab', raise_on_error=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
        
        arabic_letters = ''.join(AR_LETTERS_CHARSET)
        allowed_chars = arabic_letters + '٠١٢٣٤٥٦٧٨٩0123456789' + ' '
        self.clean_regex = re.compile(r'[^' + re.escape(allowed_chars) + r']+')
        
        self.stemmer = ISRIStemmer()
        
        response = requests.get(STOPWORDS_URL)
        self.stopwords = response.text.splitlines()
        

    def __call__(self, text: str) -> str:
        if not isinstance(text, str):
            raise TypeError("text passed to TextProcessingPipeline is not a str")
            
        text = self.clean_arabic_text(text)
        text = self.remove_emoji(text)
        
        text = self.remove_tashkeel(text)
        text = self.remove_tatweel(text)
        
        text = self.normalize_arabic(text)
        
        tokens = self.tokenize(text)
        tokens = self.remove_stopwords(tokens)
        return " ".join(tokens)
    
    def remove_tashkeel(self, text: str) -> str:
        return dediac_ar(text)

    def remove_tatweel(self, text: str) -> str:
        return re.sub(r'\u0640+', '', text)

    def clean_arabic_text(self, text: str) -> str:
        cleaned = self.clean_regex.sub('', text)
        cleaned = ' '.join(cleaned.split())
        return cleaned

    def normalize_arabic(self, text: str) -> str:
        text = re.sub(r"[\u064B-\u0652]", "", text)
        text = re.sub("[إأٱآا]", "ا", text)
        text = re.sub("ى", "ي", text)
        text = re.sub("ؤ", "ء", text)
        text = re.sub("ئ", "ء", text)
        return text

    def remove_emoji(self, text: str) -> str:
        return emoji.replace_emoji(text, replace='')
        
    def tokenize(self, text: str) -> list:
        return word_tokenize(text)
    
    def remove_stopwords(self, tokens: list) -> list:
        return [word for word in tokens if word not in self.stopwords]
    
class FeatureBuilder:
    def __init__(self, preprocessor, vectorizer):
        self.preprocessor = preprocessor
        self.vectorizer = vectorizer

    def _clean(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["cleaned_text"] = df["text"].apply(self.preprocessor)
        return df

    def fit_transform(self, df: pd.DataFrame):
        df = self._clean(df)
        return self.vectorizer.fit_transform(df["cleaned_text"]), df["sentiment"].values

    def transform(self, df: pd.DataFrame):
        df = self._clean(df)
        return self.vectorizer.transform(df["cleaned_text"]), df["sentiment"].values

    def save(self, path: str):
        joblib.dump(self.vectorizer, f"{path}.pkl")


def build_features(preprocessor=None, **tfidf_kwargs) -> FeatureBuilder:
    preprocessor = preprocessor or TextProcessingPipeline()
    return FeatureBuilder(preprocessor, TfidfVectorizer(**tfidf_kwargs))