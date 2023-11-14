#importing necessary libraries...
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re

# Loading and displying the data from csv file...
df = pd.read_csv("C:\\Users\\virkn\\Downloads\\NLP\\measuring_hate_speech.csv")

df.head(5)

#importing necessary library for preprocessing...
import string

#preprocessing of data...
def clean_and_tokenize(text):
    # Convert to lowercase
    text_lower = text.lower()
    
    # Removing punctuation
    text_no_punct = ''.join(char for char in text_lower if char not in string.punctuation)
    
    # Removing digits
    text_no_digits = ''.join(char for char in text_no_punct if not char.isdigit())
    
    # Removing extra whitespaces
    text_cleaned = ' '.join(text_no_digits.split())
    
    return text_cleaned

# Applying the preprocessing function to the 'text' column
df['cleaned_text'] = df['text'].apply(clean_and_tokenize)


# Tokenize and removing stop words
stop_words = set(stopwords.words('english'))
df['tokenized_text'] = df['cleaned_text'].apply(lambda text: [word for word in word_tokenize(text) if word not in stop_words])

# Converting tokenized text back to a string
df['processed_text'] = df['tokenized_text'].apply(lambda tokens: ' '.join(tokens))


# Building the Bag-of-Words model
vectorizer = CountVectorizer(max_features=6000) 
X = vectorizer.fit_transform(df['processed_text'])

# Convert the sparse matrix to a DataFrame
bow_df = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())


# Display the bag-of-words DataFrame
print(bow_df.head())





