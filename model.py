import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

#import dataset
dataset=pd.read_csv("Restaurant_Reviews.tsv",delimiter="\t",quoting=3)

#cleaning test
import re
import nltk
nltk.download("stopwords")
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

corpus = []
for i in range(0,1000):
    review = re.sub('[^a-zA-Z]', ' ' ,dataset['Review'][i])
    review = review.lower()
    review = review.split()
    ps=PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words("english"))]
    review = " ".join(review)
    corpus.append(review)
    
"""#sample data
    
#Term frequency and Inverse term frequency
from sklearn.feature_extraction.text import TfidfVectorizer
ti = TfidfVectorizer(max_features = 1500,min_df = 2)
out = ti.fit_transform(li).toarray()
"""    
#creating the bag of models

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
x=cv.fit_transform(corpus).toarray()
y=dataset.iloc[:,-1].values

pickle.dump(cv,open("Countvec.pkl","wb"))
pickle.dump(ps,open("portStem.pkl","wb"))



# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.20, random_state = 0)

# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)
pickle.dump(classifier,open("Gaussclass.pkl","wb"))