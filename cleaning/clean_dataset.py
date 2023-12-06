import pandas
import logging
import contractions
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import emoji


class CleanTextDatasetOperator:

    def __init__(
            self, file_path="", *args, **kwargs
    ):
        """
        This class wraps all data cleaning logic under one shade.

        Args:   
            file_path (str) : Path to the text dataset file in csv format.
        Returns:
            cleaned_df (pandas.DataFrame) : processed cleaned data frame.

        Added By : Abbas Ismail
        """
        self.file_path = file_path

    def load_csv_to_dataset(self):
        """
        This method loads CSV dataset to pandas Dataframe
        Returns:
            df : pandas.DataFrame - dataset dataframe

        Added By : Abbas Ismail
        """
        try:
            text_df = pandas.read_csv(
                filepath_or_buffer=self.file_path
            )[['hatespeech', 'text', "sentiment","respect","insult","humiliate","dehumanize","violence","genocide","attack_defend",]]

            n_features = text_df.columns
            n_len = text_df.__len__()

            logging.info(
                f"...completed dataset loaded to dataframe, features : {n_features}, \n"
                f" number of records : {n_len}"
            )
            return text_df
        except Exception as ex:
            raise Exception(
                f"Something went wrong while loading the dataset file to dataframe : {ex}"
            )
            

    def lowercase_text(self, df : pandas.DataFrame):
        """
        Method converts the dataset text to lower case

        Added By : Sai Kumar Adulla
        """
        df['text'] = df['text'].str.lower()
        return df
    
    def remove_number(self, df : pandas.DataFrame):
        """
        Method remove numericals from datatset text

        Added By : Lakshmi Kumari
        """
        def contains_numbers(cell):
            return bool(re.search(r'\d', str(cell)))

        # Identify columns with numbers
        columns_with_numbers = [col for col in df.columns if any(df[col].apply(contains_numbers))]

        # Replace numbers with NaN in identified columns
        df[columns_with_numbers] = df[columns_with_numbers].applymap(lambda x: None if contains_numbers(x) else x)

        # Drop columns with all null values
        df_cleaned = df.dropna(axis=1, how='all')

        return df_cleaned

    def expand_contractions(self, df : pandas.DataFrame):
        """
        Method for expanding compression for text column

        Added By : Simranjeet and Navneet kaur
        """
        df= df['text'].apply(contractions.fix)
        return df
    
    def remove_html_tags(self, df : pandas.DataFrame):
        """
        Method to remove all the HTML Tags from the Dataframe

        Added By : Kanika Kataria (C0866652)
        """
        def remove_html_tags(text):
            clean = re.compile('<.*?>')
            return re.sub(clean, '', text)

        df['text'] = df['text'].apply(lambda x : remove_html_tags(x))
        return df
    
    def remove_stopwords(self, df : pandas.DataFrame):
        """
        Method for removing all the english stopwords from the dataset

        Added By : Christin Paul
        """
        def _remove_words(text):
            words = word_tokenize(text)
            stop_words = set(stopwords.words('english'))
        
            processed_words = [word for word in words if word.lower() not in stop_words]
        
            processed_text = ' '.join(processed_words)
            return processed_text
        
        
        df['text'] = df['text'].apply(_remove_words)

        return df
    
    def remove_emojis(self, df : pandas.DataFrame):
        """
        Method for removing all the emojis from text

        Added By : Sharon Victor
        """
        # Apply the emoji.demojize function to the 'text' column
        df['text'] = df['text'].apply(lambda x: emoji.demojize(x))
        return df