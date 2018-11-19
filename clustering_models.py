import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
from urllib.request import Request, urlopen
import datetime
import string
from string import digits
import collections
import scipy.stats as scs
import cc_pipeline as P
import time
import random
import pickle

#sentiment and language
import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import spacy
import vaderSentiment
from langdetect import detect
from gensim.models import Word2Vec
from gensim import corpora
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.decomposition import PCA, KernelPCA
from sklearn.pipeline import Pipeline
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import NMF, LatentDirichletAllocation
from gensim.models.ldamodel import LdaModel
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import chi2
import knee_locator

#plotting
from bokeh.plotting import figure, show, output_file, output_notebook, ColumnDataSource
from bokeh.models import HoverTool, BoxSelectTool
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from PIL import Image
import pyLDAvis.sklearn
import pyLDAvis.gensim as gensimvis
import pyLDAvis

nltk.download('punkt')




def vectorize_text(doc_list):
    
    vectorizer = TfidfVectorizer(stop_words=stopwords.words('english'), lowercase=True) 
    tfidf_model = vectorizer.fit_transform(doc_list)
    
    return tfidf_model

    
def svd(doc_list):
    
    model = vectorize_text(doc_list)
    
    svd = TruncatedSVD(n_components=100, n_iter=7, random_state=42)
    clf = svd.fit_transform(model) 
    
    return clf

    
def find_elbow(doc_list):
    
    clf = svd(doc_list)
    
    distortions = []
    K = range(1,10)
    for k in K:
        kmeanModel = KMeans(n_clusters=k, n_jobs=2)
        kmeanModel.fit(clf)
        distortions.append(sum(np.min(cdist(clf, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / clf.shape[0])

    plt.plot(K, distortions, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Distortion')
    plt.title('The Elbow Method showing the optimal k')
    plt.savefig('data/kmeans_elbow.png')
    plt.show()


def kernel_pca(text, components=5):
    
    tfidf_model = vectorize_text(text)

    kpca = KernelPCA(n_components=components, kernel='rbf', gamma=15)
    X_kpca = kpca.fit_transform(tfidf_model)


    plt.figure()
    plt.title('TFIDF - KernelPCA')
    plt.scatter(X_kpca[:, 0], X_kpca[:, 1], c='rgb')
    plt.savefig('data/kernel_pca.png')
    plt.show()
    
    
def standard_pca(text, components=4):

    tfidf_model = vectorize_text(text)
    tfidf_dense = tfidf_model.todense()

    pca = PCA(n_components=components)
    data2D = pca.fit_transform(tfidf_dense)

    #this array is one dimesional so we plot using
    plt.scatter(data2D[:,0], data2D[:,1], c='rgb', alpha=0.1)
    plt.title('PCA Reduction')
    plt.savefig('data/standard_pca.png')
    plt.show() 
    
    
def latent_semantic_analysis(text, components=100):
    
    tfidf_model = vectorize_text(text)

    svd = TruncatedSVD(n_components=100, n_iter=7, random_state=42)
    clf = svd.fit_transform(tfidf_model) 

    #this array is one dimesional so we plot using
    plt.scatter(clf[:,0], clf[:,1], c='rgb', alpha=0.1)
    plt.title('Truncated SVD Reduction')
    plt.savefig('data/lsa.png')
    plt.show() 

    
    
    
    
if __name__=="__main__":
    
    df = pd.read_csv('data/testenglish.csv', encoding='utf-8')
    text = df['cleaned'].values
    text = text.tolist()
    
    