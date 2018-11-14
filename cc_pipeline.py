import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
from urllib.request import Request, urlopen
import datetime
import string
import collections
import scipy.stats as scs
import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
import vaderSentiment
from langdetect import detect
from knee_locator import KneeLocator

nltk.download('punkt')
nltk.download('stopwords')


def json_to_df(filename):

    '''converts .json file to dataframe
    input: 'filepath'
    output: pd.DataFrame'''

    data = filename
    chunks = pd.read_json(data, lines=True, chunksize=10000)

    chunks_list = []
    for i, chunk in enumerate(chunks):
        chunks_list.append(chunk)

    return pd.concat(chunks_list)


def get_recent(filename):

    '''reformats date column and adds new date column, gets entries after October 1st, writes new encoded with recent entries
    input: df
    output: df, encoded csv file saved to data/cc_recent_write.csv'''

    #df = json_to_df(filename)
    df = pd.read_csv(filename)

    date_series = pd.Series(df['created_at'].values)
    dates = date_series.apply(lambda x: x.get('$date'))

    df['date'] = pd.to_datetime(dates)
    df_recent =  df.loc[pd.to_datetime(df['date'].dt.date) > '2018-10-01'].reset_index()

    df_recent.to_csv('data/cc_recent_write.csv', encoding='utf-8', index=False)

    return df_recent


def scrape_text(df):

    '''scrapes text from .cc file @ url
    input: df
    output: list of captions from each url'''

    df_series = get_recent(df)
    series = pd.Series(df_series['url'])

    text = []
    for link in series:
        req = Request(link, headers={'User-Agent': 'Mozilla/5.0'})
        webpage = urlopen(req).read()
        webpage = webpage.decode('utf-8')
        text.append(webpage)

    return text

def add_text(filename):

    '''adds text to each entry in a new column
    input: df
    output: df'''

    df = get_recent(filename)
    extracts = scrape_text(df)
    extracts = pd.Series(extracts)
    df['text'] = extracts

    return df


def get_show_text(text_list):

    '''gets raw text without timestamp from list of strings
    input: list of strings
    output: string'''

    return "\n".join( ["\n".join( x.split("\n")[2:] ) for x in text_list.split("\n\n")] )


def get_all_text(filename):

    '''iterates through list of strings to return processed text_string
    input: string
    output: list of strings'''

    temp_texts = add_text(filename)
    raw = temp_texts['text'].values

    for doc in raw:

        doc_list = []

        for doc in raw:
            doc_list.append(get_show_text(doc))

    return doc_list


def clean_text(doc):
    '''cleans a string by removing punc, characters, digits, and len < 3
    input: string
    output: string'''

    doc = doc.split('\n')
    doc = ' '.join(doc)
    doc = doc.split('-')
    doc = ' '.join(doc)
    doc = doc.split('...')
    doc = ' '.join(doc)
    doc = word_tokenize(doc)

    stop_words = stopwords.words('english')
    punct = ('!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~♪¿’')

    a = [char for char in doc if char not in punct]
    b = [w for w in a if w not in stop_words]
    c = [w for w in b if len(w) > 2]
    d = [x for x in c if not (x.isdigit() or x[0] == '-' and x[1:].isdigit())]
    e = [l for l in d if l not in digits]

    f = ' '.join(e)
    g = f.lower()
    cleaned = str(g)

    return cleaned


class CleanAllDocs(BaseEstimator, TransformerMixin):

    '''cleans all text and creates new column in df, writes to data/
    input: df
    output: df'''

    def fit(self, filename):

        all_docs = []
        temp_doc_list = get_all_text(filename)

        for cc in temp_doc_list:
            cleaned_temp = clean_text(cc)
            all_docs.append(cleaned_temp)

        self.all_docs = all_docs
        df['cleaned'] = all_docs

    def transform(self):

        df.to_csv('data/cc_text_cleaned_write.csv', encoding='utf-8', index=False)

        return all_docs


class K_Means(BaseEstimator, TransformerMixin):

    '''kmeans clustering on DataFrame object'''

    def find_clusters(self, filename):

        '''finds optimal k for kmeans'''

        distortions = []
        K = range(1,10)

        for k in K:

            temp_corpus = corpus = clean_all_docs(filename)

            vectorizer = TfidfVectorizer(stop_words=stopwords.words('english'), lowercase=True)
            tfidf_model = vectorizer.fit_transform(all_docs)

            svd = TruncatedSVD(n_components=100, n_iter=7, random_state=42)
            temp_clf = svd.fit_transform(tfidf_model)

            kmeanModel = KMeans(n_clusters=k)
            kmeanModel.fit(temp_clf)
            distortions.append(sum(np.min(cdist(temp_clf, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / temp_clf.shape[0])

        # Plot the elbow
        knee = KneeLocator(K, distortions, curve='convex', direction='decreasing')
        knee.plot_knee_normalized()
        plt.xlabel('k')
        plt.ylabel('Distortion')
        plt.title('The Elbow Method showing the optimal k')

        return self.knee.knee

    def fit(self, filename):

        corpus = clean_all_docs(filename)
        vector = TfidfVectorizer(stop_words='english')
        self.vector = vector
        matrix = vector.fit_transform(corpus)

        svd = TruncatedSVD(n_components=100, n_iter=7, random_state=42)
        clf = svd.fit_transform(matrix)
        self.clf = clf

        num_clusters = self.knee.knee
        km = KMeans(n_clusters=num_clusters)
        self.km = km.fit(self.clf)

        return self

    def transform(self):

        pca = PCA(n_components=2).fit(self.clf)
        self.reduced = pca.transform(self.clf)
        self.centroids = pca.transform(self.km.cluster_centers_)

        return self

    def plot_kmeans(self):

        plt.scatter(self.reduced[:,0], data2D[:,1], c=clusters, alpha=0.5)
        plt.scatter(self.centroids[:,0], centers2D[:,1],
            marker='x', s=200, linewidths=3, c='r')
        plt.title('(REDUCED) Closed Caption Kmeans Clusters')
        plt.show()


def pipeline_grid_search(X_train, y_train, pipeline, params, scoring):
    '''
    Runs a grid search on the given pipeline with the given params
    Parameters:
    --------------------------
    X_train  : 2 dimensional array-like
    y_train  : 1 dimensional array-like
    pipeline : Sklearn pipeline object
    params   : dictionary of pipeline parameters
    Returns:
    --------------------------
    the resulting GridSearchCV object
    '''
    grid = GridSearchCV(pipeline, params, scoring=scoring, n_jobs=-1, cv=5)
    grid.fit(X_train, y_train)

    return grid
