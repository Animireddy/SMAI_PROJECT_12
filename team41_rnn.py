#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd
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


df = df[['genre','lyrics']]


# In[4]:


# print(df2.groupby('genre').count())
# print(df11.count())
# df = df11
df = df.dropna()
# df = df.loc[df['genre'].isin(['Country','Electronic','Hip-Hop','Jazz','Metal','Pop','Rock','Folk'])]
df = df.loc[df['genre'].isin(['Country','Hip-Hop','Metal','Pop','Rock'])]
df.count()
print(df.groupby('genre').count())


# In[5]:


df.info()


# In[6]:


text = df['lyrics'][0]
text = '\t'.join(text.split("\n"))
text = ' '.join(text.split())
print(text)


# In[7]:


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


# In[8]:


df['lyrics'][0]


# In[9]:


# The maximum number of words to be used. (most frequent)
# The maximum number of words to keep, based on word frequency. Only the most common num_words-1 words will be kept
MAX_NB_WORDS = 50000
# Max number of words in each song. #vocabulary top 250 frequent words
MAX_SEQUENCE_LENGTH = 250
# This is fixed.
EMBEDDING_DIM = 100

tokenizer = Tokenizer(num_words=MAX_NB_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
tokenizer.fit_on_texts(df['lyrics'].values)
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))


# In[10]:


X = tokenizer.texts_to_sequences(df['lyrics'].values)
X = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)
print('Shape of data tensor:', X.shape)


# In[11]:


Y = pd.get_dummies(df['genre']).values
print('Shape of label tensor:', Y.shape)


# In[ ]:




