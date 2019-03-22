#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from nltk.stem.porter import *
from imblearn.over_sampling import RandomOverSampler

import matplotlib.pyplot as plt
import seaborn as sns
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.layers import Dropout
import re
from nltk.corpus import stopwords
from nltk import word_tokenize
STOPWORDS = set(stopwords.words('english'))
from bs4 import BeautifulSoup
import plotly.graph_objs as go
import plotly.plotly as py


# In[2]:


df = pd.read_csv('./corpus/lyrics.csv')
df.head()


# In[3]:


# print(df2.groupby('genre').count())
# print(df11.count())
# df = df11
df = df[0:10000]
df = df[['genre','lyrics']]
df = df.dropna()
# df = df.loc[df['genre'].isin(['Country','Electronic','Hip-Hop','Jazz','Metal','Pop','Rock','Folk'])]
df = df.loc[df['genre'].isin(['Country','Hip-Hop','Metal','Pop','Rock'])]
df.count()
print(df.groupby('genre').count())


# In[4]:


df.info()


# In[5]:


text = df['lyrics'][0]
text = '\t'.join(text.split("\n"))
text = ' '.join(text.split())
print(text)


# In[6]:


df = df.reset_index(drop=True)
REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;?]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
STOPWORDS = set(stopwords.words('english'))

def clean_text(text):
    """
        text: a string
        
        return: modified initial string
    """
    text = '\t'.join(text.split("\n"))
    text = ' '.join(text.split())
    text = text.lower() # lowercase text
    text = REPLACE_BY_SPACE_RE.sub(' ', text) # replace REPLACE_BY_SPACE_RE symbols by space in text. substitute the matched string in REPLACE_BY_SPACE_RE with space.
    text = BAD_SYMBOLS_RE.sub('', text) # remove symbols which are in BAD_SYMBOLS_RE from text. substitute the matched string in BAD_SYMBOLS_RE with nothing. 
#    text = re.sub(r'\W+', '', text)
    text = ' '.join(word for word in text.split() if word not in STOPWORDS) # remove stopwors from text
    return text
df['lyrics'] = df['lyrics'].apply(clean_text)


# In[7]:


X = df['lyrics']
Y = df['genre']

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.10, random_state = 42)


# In[ ]:


stemmer = PorterStemmer()
f = open('output.txt', 'w')

def stemmed_words(doc):
    return (stemmer.stem(w) for w in analyzer(doc))

analyzer = CountVectorizer().build_analyzer()
print("Creating Word vectors\n")
vectorizer = CountVectorizer(analyzer=stemmed_words,
                             tokenizer=None,
                             lowercase=True,
                             preprocessor=None,
                             max_features=5000
                             )

train_data_features = vectorizer.fit_transform([r for r in X_train])
# print(train_data_features) dimensions
test_data_features = vectorizer.transform([r for r in X_test])

train_data_features = train_data_features.toarray()
test_data_features = test_data_features.toarray()
# print(train_data_features) zero matrices with corresponding dimensions
# print(test_data_features)
# print("Resampling corpus\n")
# rs = RandomOverSampler()
# X_resampledRe, y_resampledRe = rs.fit_sample(train_data_features,Y_train)

print("fitting for SVC\n")
clf = SVC()
clf.fit(train_data_features, Y_train)
f.write("\nOutput from SVC Normal:\n")
predicted = clf.predict(test_data_features)
f.write(str(predicted))
f.write("\naccuracy: ")
f.write(str(np.mean(predicted == Y_test)))
f.write("\n")


# In[ ]:




