import nltk
import string

import numpy as np 
import pandas as pd 

import os
from pathlib import Path

import string
import nltk
from nltk.corpus import stopwords

import scipy.io
import scipy.linalg
from scipy.sparse import csr_matrix, vstack, lil_matrix 
from sklearn.base import TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import MultinomialNB

import plotly.express as px
import plotly.figure_factory as ff
from yellowbrick.text import TSNEVisualizer
