from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas
import logging

class FeatureSelectDatasetOperator:
    def __init__(
            self, cleaned_dataframe : pandas.DataFrame, *args, **kwargs
    ):
        """
        This class wraps all feature selection logic under one shade.

        Args:   
            file_path (str) : Path to the text dataset file in csv format.
        Returns:
            feature_df (pandas.DataFrame) : gives data frame words to train the model.

        Added By : Sai Kumar Adulla
        """
        self.dataset_df = cleaned_dataframe

    def create_bow_matrix(self, df : pandas.DataFrame, input_col='text', output_col='bow_features', max_features=5000):
        """
        The CountVectorizer is used to transform the input DataFrame and generate the BoW features

        Added By :  Christin Paul
        """
        vectorizer = CountVectorizer(inputCol=input_col, outputCol=output_col, vocabSize=max_features)

        bow_df = vectorizer.fit(df).transform(df)

        return bow_df

    def tfidf(self, df : pandas.DataFrame):
        """
        fit_transform method to the specified text column and creates a DataFrame with the TF-IDF matrix.

        Added By :  Sai Kumar Adulla
        """
        tfidf_vectorizer = TfidfVectorizer(max_features=5000)  

        tfidf_matrix = tfidf_vectorizer.fit_transform(df['text'])
        
        tfidf_df = pandas.DataFrame(tfidf_matrix.toarray(), columns=tfidf_vectorizer.get_feature_names_out())
        
        return tfidf_df

    def feature_creation(self, df : pandas.DataFrame):
        """
        Creatiing features

        Added By :  Sharon Victor
        """
        # 'Score' feature for 'hate_speech_score' column
        df['Score'] = df['hatespeech'].astype(float)

         # 'Comments' feature for 'text' column
        df['Comments'] = df['text'].str.len()

        return df