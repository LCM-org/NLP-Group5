import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


def remove_stopwords(text):
    words = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
  
    processed_words = [word for word in words if word.lower() not in stop_words]
  
    processed_text = ' '.join(processed_words)

    return processed_text
