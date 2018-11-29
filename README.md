# closed_caption_classifier
TV Captions - Unsupervised Machine Learning and NLP (in progress)

Youtube: https://www.youtube.com/watch?v=my8ZuVVUFoI

Overview:
Sentiment analysis and unsupervised classification of TV shows and top brands based on closed caption text.

Scope:
From opinion polls and product reviews to shaping market strategy, NLP is a tool with the ability to transform businesses. The scope of this project is to create a clustering model and to provide brand awareness and sentiment analysis to top brands and TV broadcast stations based on their content. With this information companies will be better informed of latent brand perception and take action if necessary.  

Tools/Environments:
The main tools used in this project were AWS, Jupyter Notebook, and Python, specifically the scikit learn library, spaCy NLP library, langdetect for language separation, and pyLDAviz visualization library. A pilot model was created locally and scaled up on an AWS EC2 instance.

Data Scraping/Cleaning/Wrangling:
Data was scraped from text files stored on an AWS S3 bucket using Python’s Requests library . A script was created using User:Agent and sleep timing to passively scrape files and prevent IP blocking from AWS. 

Scraped text was then pre-processed into two separate formats, one for sentiment scoring and another for modeling. Modeling data was first tokenized/lemmatized and then the TF-IDF for n_gram range (1-3) calculated. TF-IDF created a matrix of weight vectors assigned to each word allowing important words to have a higher weight and decreasing the weight of commonly used words.  

A number of dimensionality reduction and topic modeling techniques were applied including Kernel PCA, and SVD prior to improve clustering and SVD was found to be the best performing method. An LDA model was then used to automatically discover topics which were common among all captions.

Captions were then clustered via a KMeans algorithm where the optimal number of clusters, K, was found using the elbow method. This method is a KMeans clustering on the dataset for a range of values of K where and each value of K is then plotted along with the calculated sum of squared errors (SSE) and the optimal K is the inflection point with a low K and reduced SSE. Clusters and topics were then visualized using Python’s pyLDAviz library as well as matplotlib.

Python's VADER library was used to score and list the sentiment of a list of 175 top brands.

Results:
Results from 50,000 captions with K=15 clusters and LDA for 20 topics shows separation of TV shows into distinct clusters and topics. 

