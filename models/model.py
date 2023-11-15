import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import confusion_matrix
import pandas



class TrainHateDetectionModel:

    def __init__(
        self, dataset : pandas.DataFrame, test_size = 0.1,
    ):
        """
        This class trains multiple ML models for the 
        detecting hate speech and yields list of evalution metrics and prediction methods
        Args:
            dataset (pandas.DataFrame) - dataset having text and label fields.

        Added by : Abbas Ismail
        """
        self.dataset = dataset
        self.test_size = test_size

    def prepare_test_train_data(self):
        """
        This method splits the dataset into train and test groups
        Returns:
            df : pandas.DataFrame - dataset dataframe

        Added By :Jenny Joshi
        """

        # Split the dataset into training and testing sets
        self.train_data, self.test_data, self.train_labels, self.test_labels = train_test_split(
            self.dataset['text'],
            self.dataset['label'],
            test_size=0.2,
            random_state=42
        )

    def buildNavieBayesModel(self):
        """
        Builds and Trains the Naive Bayes Model for HateDetection

        Added By :Abbas Ismail
        """

        # Feature Engineering: Use TF-IDF Vectorizer
        vectorizer = CountVectorizer()
        self.X_train = vectorizer.fit_transform(self.train_data)
        self.X_test = vectorizer.transform(self.test_data)

        # Optionally, you can add more feature engineering steps here:
        # For example, scaling numerical features, dimensionality reduction, etc.

        # Train a Naive Bayes classifier
        self.naive_bayes_classifier = MultinomialNB()
        self.naive_bayes_classifier.fit(self.X_train, self.train_labels)

    def getTestPrecitions(self):
        """
        Run the predcitions on the test sets

        Added By :Sai Kumar Adulla
        """

        # Make predictions on the test set
        self.predictions = self.naive_bayes_classifier.predict(self.X_test)

    def yield_model_accuracy_metrics(self):

        """
        Method prints the model performance metrics
            - accuracy_score
            - classification_report
            - confusion matrix

        Added By : Christin Paul
        """

        # Evaluate the model
        accuracy = accuracy_score(self.test_labels, self.predictions)
        print(f'Accuracy: {accuracy:.2f}')

        # Display classification report
        print(classification_report(self.test_labels, self.predictions))

        # Confusion matrix
        conf_matrix = confusion_matrix(self.test_labels, self.predictions)
        print("Confusion Matrix:\n", conf_matrix)