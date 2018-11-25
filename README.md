# closed_caption_classifier
Classification of TV closed-caption data (in progress)

Overview:
Sentiment analysis and unsupervised classification of TV shows and top brands based on closed caption text.

Scope:
From opinion polls and product reviews to shaping market strategy, NLP is a tool with the ability to transform businesses. The scope of this project is to create a classification model to provide brand awareness and sentiment analysis to top brands and TV broadcast stations based on their content. With this information companies will be better informed of latent brand perception and take action if necessary. 

Constraints and assumptions:
One major assumption is that each show will have varying content allowing for distinct classification. Another major assumption is that the raw pos/neg polarity score is loosely representative of the sentiment of the entire text, and finally that the sentiment of a brand could be associated with the polarity score of its associated TV show. 

Tools/Environments:
The main tools used in this project were AWS, Jupyter Notebook, and Python, specifically the scikit learn library, spaCy NLP library, langdetect for language separation, and pyLDAviz visualization library. A pilot model was created locally and scaled up on an AWS EC2 instance.

Data Scraping/Cleaning/Wrangling:
Data was scraped from text files stored on an AWS S3 bucket using Python’s Requests library . A script was created using User:Agent spoofing and rotating proxies to passively scrape files and prevent IP blocking from Amazon. 

Scraped text was then pre-processed in two separate formats, one for sentiment scoring and another for modeling. Modeling data was first tokenized/lemmatized and then the TF-IDF for n_gram range (1-3) calculated. TF-IDF created a matrix of weight vectors assigned to each word allowing important words to have a higher weight and decreasing the weight of commonly used words. 

Dimensionality reduction is the process of converting data of very high dimensionality, in this case the vectorized matrix, into data of a much lower dimensionality such that each of the lower dimensions convey much more information. 

A number of dimensionality reduction and topic modeling techniques were applied including Kernel PCA, SVD, and LDA prior to improve clustering and, after using a GridSearch to investigate each model with varying parameters, LDA was found to be the best performing model. 

LDA is a modeling technique used to automatically discover topics which are latent among a group of documents, in this case each TV show caption.

Captions were then clustered and classified via a KMeans algorithm where the optimal number of clusters, K, was found using the elbow method. This method is a KMeans clustering on the dataset for a range of values of K where and each value of K is then plotted along with the calculated sum of squared errors (SSE) and the optimal K is the inflection point with a low K and reduced SSE.

Clusters and topics were then visualized using Python’s pyLDAviz library as well as matplotlib.

Results:
Results from 50,000 captions and using LDA and K=25 clusters shows separation of TV shows into distinct topics. 

