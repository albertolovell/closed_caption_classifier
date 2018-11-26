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
from pprint import pprint
from collections import Counter

#sentiment and language
import nltk
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import spacy
import vaderSentiment
from langdetect import detect
from gensim.models import Word2Vec
from gensim import corpora
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from spacy import displacy



def inline_text(show_raw):
    
    '''returns show text without timestamps'''
    
    temp = " ".join( ["\n".join( x.split("\n")[2:] ) for x in show_raw.split("\n\n")] )
    temp = temp.split('\n')
    temp = " ".join(temp)
    return temp


def sent_for_spacy(text_list):
    
    '''cleans all text and creates new column in dataframe'''
    
    doc_list = []
    for doc in text_list:
        cleaned = inline_text(doc)
        tok = sent_tokenize(cleaned)
        doc_list.append(tok)
    return doc_list


def get_sentiment_sentence(sent_tok, brands):
    
    analyser = SentimentIntensityAnalyzer()
    scores = []
    
    for sents in sent_tok:
        for brand in brands:
            for sent in sents:
                if brand in sent:
                    score = list(dict.items(analyser.polarity_scores(sent)))
                    scores.append([brand, score])
            
    return scores




if __name__=="__main__":
    
    with open ('data/short_brands.pkl', 'rb') as r:
        brands = pickle.load(r)
        
    df = pd.read_csv('data/clean_english_stations.csv', encoding='utf=8')
    text_series = df['text'].values
    sent_tok = sent_for_spacy(text_series)
    
    with open ('data/all_docs_sent_tokenized.pkl', 'wb') as f:
        pickle.dump(sent_tok, f)
        
    sentiment = get_sentiment_sentence(sent_tok, brands)
    with open ('data/sentiments.pkl', 'wb') as f:
        pickle.dump(sentiment, f)
    






