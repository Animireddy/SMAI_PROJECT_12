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




