# -*- coding: utf-8 -*-
"""
Created on Thu Mar  1 14:09:43 2018

@author: Karthikeyan
"""
import pandas as pd
import numpy as np


DATA_FILE = 'E:\\Internship\\Sentiment_analysis\\train.csv'
df = pd.read_csv(DATA_FILE,encoding='utf-8')
print(df.head())

Category = df.categories
Conversation = df.converse.astype(str)



from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalMaxPooling1D
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
import time
from keras import metrics

num_max = 1000
# preprocess
le = LabelEncoder()
Category = le.fit_transform(Category)
Category = np_utils.to_categorical(Category)
tok = Tokenizer(num_words=num_max)
tok.fit_on_texts(Conversation)
mat_texts = tok.texts_to_matrix(Conversation,mode='count')
print(Category[:5])
print(mat_texts[:5])
print(Category.shape,mat_texts.shape)

# try a simple model first

def get_simple_model():
    model = Sequential()
    model.add(Dense(512, activation='relu', input_shape=(num_max,)))
    model.add(Dropout(0.2))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(6, activation='softmax'))
    model.summary()
    model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
    print('compile done')
    return model

def check_model(model,x,y):
    model.fit(x,y,batch_size=32,epochs=25,verbose=1,validation_split=0.2)

m = get_simple_model()
check_model(m,mat_texts,Category)

# for cnn preproces
max_len = 100
cnn_texts_seq = tok.texts_to_sequences(Conversation)
print(cnn_texts_seq[0])
cnn_texts_mat = sequence.pad_sequences(cnn_texts_seq,maxlen=max_len)
print(cnn_texts_mat[0])
print(cnn_texts_mat.shape)

##cnn model 1
def get_cnn_model_v1():   
    model = Sequential()
    # we start off with an efficient embedding layer which maps
    # our vocab indices into embedding_dims dimensions
    # 1000 is num_max
    model.add(Embedding(1000,
                        20,
                        input_length=max_len))
    model.add(Dropout(0.2))
    model.add(Conv1D(64,
                     3,
                     padding='valid',
                     activation='relu',
                     strides=1))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(256))
    model.add(Dropout(0.2))
    model.add(Activation('relu'))
    model.add(Dense(6))
    model.add(Activation('softmax'))
    model.summary()
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy',metrics.categorical_accuracy])
    return model

m = get_cnn_model_v1()
check_model(m,cnn_texts_mat,Category)

#cnn model 2
def get_cnn_model_v2():    # added filter
    model = Sequential()
    # we start off with an efficient embedding layer which maps
    # our vocab indices into embedding_dims dimensions
    # 1000 is num_max
    model.add(Embedding(1000,
                        20,
                        input_length=max_len))
    model.add(Dropout(0.2))
    model.add(Conv1D(256, #!!!!!!!!!!!!!!!!!!!
                     3,
                     padding='valid',
                     activation='relu',
                     strides=1))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(256))
    model.add(Dropout(0.2))
    model.add(Activation('relu'))
    model.add(Dense(6))
    model.add(Activation('softmax'))
    model.summary()
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy',metrics.categorical_accuracy])
    return model

m = get_cnn_model_v2()
check_model(m,cnn_texts_mat,Category)




























































