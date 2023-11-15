from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd

def create_bow_matrix(df):
    # Create a CountVectorizer instance
    vectorizer = CountVectorizer(max_features=5000)
    texts = df['text'].tolist()
    bow_matrix = vectorizer.fit_transform(texts)
    bow_df = pd.DataFrame(bow_matrix.toarray(), columns=vectorizer.get_feature_names_out())

    # Return the BoW DataFrame
    return bow_df

# Read your DataFrame from the CSV file
df = pd.read_csv("measuring_hate_speech.csv")

# Call the function and display the BoW DataFrame
bow_matrix_df = create_bow_matrix(df)
print(bow_matrix_df)
