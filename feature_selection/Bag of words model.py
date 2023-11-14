from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd

#Reading csv file 
df = pd.read_csv(r"C:\Users\Owner\OneDrive\Documents\measuring_hate_speech.csv")

def create_bow_matrix(texts):
    # Create a CountVectorizer instance
    vectorizer = CountVectorizer(max_features=6000)

    # Fit and transform the texts to get the BoW representation
    bow_Matrix = vectorizer.fit_transform(texts)

    # Convert the BoW matrix to a DataFrame for better visualization
    bow_df = pd.DataFrame(bow_Matrix.toarray(), columns=vectorizer.get_feature_names_out())

    # Return the BoW DataFrame
    return bow_df

# Assuming your dataset has a column named 'text'
texts = df['text'].tolist()

# Call the function to create the BoW matrix
bow_matrix_df = create_bow_matrix(texts)

# Display the BoW DataFrame
print(bow_matrix_df)
