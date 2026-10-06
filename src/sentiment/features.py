import re

import emoji
import nltk
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
    


preprocessor = TextProcessingPipeline()
vectorizer = TfidfVectorizer()
