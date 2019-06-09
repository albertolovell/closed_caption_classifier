# Unsupervised Machine Learning and NLP 

### Scope and Overview:

This dataset is a collection of over 50M unlabeled closed caption text files acquired by a Bay Area entertainment company. 

The scope of this project is to create a clustering model for categorization of all unlabeled closed captions, Topic Modelling to discover hidden topics within the dataset, and to provide sentiment analysis to 175 top brands based on their product placement within television broadcasts. Results from this workflow will have downstream applications in content recommendation systems and auto-tagging.

### Tools & Environments:

The main tools used in this project were AWS (EC2), Jupyter Notebook, and Python, specifically the scikit learn library, spaCy NLP library, langdetect for language separation, and pyLDAviz visualization library.

### Workflow:

Caption data was scraped from text files stored on an AWS S3 bucket using Python’s Requests library and a script was created using User:Agent and sleep timing to passively scrape files to prevent IP blocking from AWS.  

Text was then TFIDF vectorized and clustered via a KMeans algorithm and categorized accordingly. The optimal number of clusters, K, was estimated by tracking the reduced sum of squared error with increasing cluster number and then confirmed. A Latent Dirichlet Allocation (LDA) model was then used to automatically discover hidden topics which were common among all captions.

Finally, Python's VADER library was used to score and list the sentiment of a list of 175 top brands.