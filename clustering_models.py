import pandas as pd
import numpy as np
import scipy.stats as scs
import cc_pipeline as P

import nltk
import spacy
import vaderSentiment
from langdetect import detect
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import NMF, LatentDirichletAllocation




def vectorize_text(doc_list):
    
    vectorizer = TfidfVectorizer(stop_words=stopwords.words('english'), lowercase=True) 
    tfidf_model = vectorizer.fit_transform(doc_list)

