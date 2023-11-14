import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

df = pd.read_csv('measuring_hate_speech.csv')

# This result df won't give the actual dataframe 
# This is used to convert a collection of text documents into numerical vectors 
def tfidf(df):
    tfidf_vectorizer = TfidfVectorizer(max_features=5000)  
    tfidf_matrix = tfidf_vectorizer.fit_transform(df['text'])
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_vectorizer.get_feature_names_out())
    print("DataFrame with TF-IDF values:")
    print(tfidf_df)

tfidf(df)