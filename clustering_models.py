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
from collections import Counter

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
from spacy import displacy

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
import umap

nltk.download('punkt')




def vectorize_text(doc_list):
    
    vectorizer = TfidfVectorizer(stop_words=stopwords.words('english'), ngram_range=(1, 3), lowercase=True) 
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
    
def plot_kmeans(df, text, clusters=5):
    
    temp_df = df
    
    vectorizer = TfidfVectorizer(stop_words=stopwords.words('english'), ngram_range=(1, 3), lowercase=True) 
    tfidf_model = vectorizer.fit_transform(text)
    
    kmeans = KMeans(n_clusters=clusters, n_init = 5, n_jobs = -1)
    km = kmeans.fit_transform(tfidf_model)
    
    #common words amongst each cluster
    common = []
    words = vectorizer.get_feature_names()
    common_words = kmeans.cluster_centers_.argsort()[:,-1:-11:-1]
    for num, centroid in enumerate(common_words):
        common.append(str(num) + ' : ' + ', '.join(words[word] for word in centroid))
    
    kmeans_df = temp_df
    kmeans_df['cluster'] = kmeans.labels_
    kmeans_df.to_csv('data/all_kmeans_clusters.csv', encoding='utf-8', index=False)
    
    return common
    


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
    
    
def latent_semantic_analysis(text, components=5):
    
    tfidf_model = vectorize_text(text)

    svd = TruncatedSVD(n_components=components, n_iter=7, random_state=42)
    clf = svd.fit_transform(tfidf_model) 

    #this array is one dimesional so we plot using
    plt.scatter(clf[:,0], clf[:,1], c='rgb', alpha=0.1)
    plt.title('Truncated SVD Reduction')
    plt.savefig('data/lsa.png')
    plt.show() 
    
def show_important_words(text):
    
    vectorizer = TfidfVectorizer(stop_words=stopwords.words('english'), ngram_range=(1, 3), lowercase=True) 
    tfidf_model = vectorizer.fit_transform(text)
    
    svd = TruncatedSVD(n_components=100, n_iter=7, random_state=42)
    clf = svd.fit_transform(tfidf_model) 
    
    terms = vectorizer.get_feature_names()

    for i, comp in enumerate(svd.components_):
    
        terms_comp = zip(terms, comp)
        sorted_terms = sorted(terms_comp, key= lambda x:x[1], reverse=True)[:7]
        print("Topic "+str(i)+": ")
        for t in sorted_terms:
            print(t[0])
            print(" ")
    
    return
    
def plot_topics(text, components=5):
    
    vectorizer = TfidfVectorizer(stop_words=stopwords.words('english'), ngram_range=(1, 3), lowercase=True) 
    tfidf_model = vectorizer.fit_transform(text)
    svd = TruncatedSVD(n_components=components, n_iter=7)
    clf = svd.fit_transform(tfidf_model) 
    
    X_topics = svd.fit_transform(tfidf_model)
    embedding = umap.UMAP(n_neighbors=150, min_dist=0.5, random_state=12).fit_transform(X_topics)

    plt.scatter(embedding[:, 0], embedding[:, 1], 
    c = 'rgb',
    s = 10, # size
    edgecolor='none'
    )
    plt.show()
    
    
def plot_wordcloud(text_string):
    
    wordcloud = WordCloud().generate(text_string)

    plt.figure(figsize=(10, 30))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title('Topic 0')
    plt.axis("off")
    plt.show()
    

    
    
    
if __name__=="__main__":
    
    df = pd.read_csv('filepath/filename', encoding='utf-8')
    text = df['cleaned'].values
    text = text.tolist()
    
    #pipeline moving forward
    
    