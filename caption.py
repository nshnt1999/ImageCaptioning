#!/usr/bin/env python
# coding: utf-8

# In[14]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import keras
import json
from keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from keras.preprocessing import image
from keras.models import Model, load_model
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Input, Dense, Dropout, Embedding, LSTM
from keras.layers.merge import add
import pickle


# In[2]:


model_conv = ResNet50(weights="imagenet",input_shape=(224,224,3))
model_conv.summary()

# In[3]:


model_new = Model(model_conv.input,model_conv.layers[-2].output)
#model_new._make_predict_function()


# In[8]:


model=load_model('./model_weights/model_9.h5')


# In[9]:


def preprocess_img(img):
    img = image.load_img(img,target_size=(224,224))
    img = image.img_to_array(img)
    img = np.expand_dims(img,axis=0)  ### bascially reshaping it(adding a 4th dimension so shape gets(1,224,224,3) )
    # Normalisation that is done by resnet50 so we have to do it in preprocessing
    img = preprocess_input(img)
    return img


# In[10]:


def encode_image(img):
    img = preprocess_img(img)
    feature_vector = model_new.predict(img)
    
    feature_vector = feature_vector.reshape((1,-1))
    #print(feature_vector.shape)
    return feature_vector


# In[19]:


with open('./saved/word_to_idx.pkl','rb') as f:
    word_to_idx = pickle.load(f)
    
with open('./saved/idx_to_word.pkl','rb') as f:
    idx_to_word = pickle.load(f)


# In[20]:


def predict_caption(photo):
    
    max_len = 35
    photo = encode_image(photo).reshape((1,2048))
    in_text = "startseq"
    for i in range(max_len):
        sequence = [word_to_idx[w] for w in in_text.split() if w in word_to_idx]
        sequence = pad_sequences([sequence],maxlen=max_len,padding='post')
        
        ypred = model.predict([photo,sequence])
        ypred = ypred.argmax() #WOrd with max prob always - Greedy Sampling There is another kind os sa
        word = idx_to_word[ypred]
        in_text += (' ' + word)
        
        if word == "endseq":
            break
    
    final_caption = in_text.split()[1:-1]
    final_caption = ' '.join(final_caption)
    return final_caption


# In[ ]:





# In[ ]:




