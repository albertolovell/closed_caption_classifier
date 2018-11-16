import numpy as np
import pandas as pd
import string
from string import digits
import collections
from langdetect import detect

nltk.download('punkt')
nltk.download('stopwords')



def get_show_text(show_raw):
    
    '''returns show text without timestamps'''
    
    return "\n".join( ["\n".join( x.split("\n")[2:] ) for x in show_raw.split("\n\n")] )



def clean_all_text(text_list):
    
    '''cleans all text and creates new column in dataframe'''
    
    temp_docs = temp_df['text'].values
    temp_docs = temp_docs.tolist()
    
    doc_list = []
    for doc in text_list:

        doc_list = []

        for doc in text_list:
            doc_list.append(get_show_text(doc))
    return doc_list


def clean_text(doc_string):
    '''cleans and lemmatizes a string by removing punc, characters, digits, and len(words) < 3'''
    
    stop_words = stopwords.words('english')
    punct = ('!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~♪¿’')
    remove_digits = str.maketrans('', '', digits)
    wordnet_lemmatizer = WordNetLemmatizer()
    lemmatized = []
    
    doc = doc.split('\n')
    doc = ' '.join(doc)
    doc = doc.split('-')
    doc = ' '.join(doc)
    doc = doc.split('...')
    doc = ' '.join(doc)
    doc = word_tokenize(doc)

    a = [char for char in doc if char not in punct]
    b = [w for w in a if w not in stop_words] 
    c = [w for w in b if len(w) > 3]
    d = [x for x in c if not (x.isdigit() or x[0] == '-' and x[1:].isdigit())]

    e = ' '.join(d)
    f = e.lower()
    g = f.translate(remove_digits)
    cleaned = str(g)
    doc = word_tokenize(cleaned)
    
    for word in doc:
        doc_temp = wordnet_lemmatizer.lemmatize(word)
        lemmatized.append(doc_temp)
    doc = ' '.join(lemmatized)
    
    return doc

def clean_and_return(docs_list):
    
    docs = []
    for cc in docs_list:
        cleaned_temp = clean_text(cc)
        docs.append(cleaned_temp)
        
    return docs


def lang_detect(doc_series):
    
    lang = []
    for x in doc_series:
        eng = 'en'
        span = 'es'

        try:
            if detect(x) == eng:
                lang.append(eng)
            else:
                lang.append(span)
        except:
            lang.append(None)
            
    return lang

if __name__=="__main__":
    
    temp_df = pd.read_csv('data/cc_head_text', encoding='utf-8')
    temp = temp_df['text'].values
    temp = temp.tolist()
    docs_list = clean_all_text(temp)
    cleaned_list = clean_and_return(docs_list)
    temp_df['cleaned'] = cleaned_list
    
    doc_series = pd.Series(temp_df['cleaned'].values)
    language = lang_detect(doc_series)
    temp_df['language'] = language
    english = temp_df[temp_df['language'] == 'en']
    spanish = temp_df[temp_df['language'] == 'es']
    
    english.to_csv('data/cc_20k_english.csv', encoding='utf-8', index=False)
    spanish.to_csv('data/cc_20k_spanish.csv', encoding='utf-8', index=False)
