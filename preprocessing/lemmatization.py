
#Kanika Kataria - c0866652
#Preprocessing - Lemmatization
import pandas as pd
import gensim
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS

path = "C:\\Users\\katar\\OneDrive\\Desktop\\Sem 3 - Imp\\NLP - Wed\\Project\\measuring_hate_speech.csv"

df = pd.read_csv(path)

def lemmatize(token):
    return WordNetLemmatizer().lemmatize(token, pos='v')

def tokenize(text):
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS \
                and len(token) > 2:  # drops words with less than 3 characters
            result.append(lemmatize(token))
    return result

def tokenize_and_lemmatize(df, column):
    df[column] = df[column].apply(lambda x: tokenize(x))
    df.text = df.text.apply(lambda x: str(x)[1:-1])

tokenize_and_lemmatize(df, 'text')
df.text.head(50)