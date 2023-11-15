import pandas as pd
import gensim
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS

class PreprocessDatasetoperator:

    def __init__(
            self, cleaned_dataframe : pandas.DataFrame, *args, **kwargs
    ):
        """
        This class wraps all data preprocessing logic under one shade.

        Args:   
            file_path (str) : Path to the text dataset file in csv format.
        Returns:
            cleaned_df (pandas.DataFrame) : processed cleaned data frame.

        Added By : Christin Paul
        """
        self.dataset_df = cleaned_dataframe

    def lemmatize(self, token):

        """
        Create an instance of WordNetLemmatizer and lemmatize the token as a verb

        Added By : Kanika Kataria (C0866652)
        """
        
        return WordNetLemmatizer().lemmatize(token, pos='v')

    def tokenize(self, text):

        """
        Tokenize the input text, remove stopwords, and keep tokens with length greater than 2

        Added By : Kanika Kataria (C0866652)
        """
        
        result = []
        
        for token in simple_preprocess(text):
            
            if token not in STOPWORDS and len(token) > 2:  # drops words with less than 3 characters
                result.append(self.lemmatize(token))
                
        return result

    def tokenize_and_lemmatize(self, df : pandas.DataFrame , column = 'text'):

        """
        Apply tokenization to the specified column
        Apply lemmatization to each token and join them back into a string

        Added By : Kanika Kataria (C0866652)
        """
        
        df[column] = df[column].apply(lambda x: self.tokenize(x))
        
        df[column] = df[column].apply(lambda x: ' '.join([self.lemmatize(token) for token in x]))

