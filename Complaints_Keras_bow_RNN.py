
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
import nltk
nltk.download('wordnet')
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from keras.callbacks import ModelCheckpoint
#STOPWORDS = set(stopwords.words('english'))

vocab_size = 5000
embedding_dim = 64
max_length = 200
epochs= 50
batch_size=1024
trunc_type = 'post'
padding_type = 'post'
oov_tok = '<OOV>'



data = pd.read_csv("/home/mb/projects/public/afarukcakmak/case_study_data.csv")

cdata = pd.read_csv("/home/mb/projects/public/afarukcakmak/processed_data_new.csv",header = None)
cdatas=cdata[:][1].str.replace(r'[^a-zA-Z\s]+|x{2,}', '')

I=cdatas.isnull()
cdatas[I]='money'
data.text=cdatas


skf = StratifiedKFold(n_splits=10, shuffle = True,random_state=0)

tmp=skf.split(np.zeros(len(data['product_group'])), data['product_group'])
idx_list=[idx for idx in tmp]


testid=idx_list[0][1]
valid=idx_list[1][1]

trainid=idx_list[2][1]
for i in range(3,10):
    trainid=np.hstack((trainid,idx_list[i][1]))


train_posts = data['text'][trainid]
train_tags = data['product_group'][trainid]

val_posts = data['text'][valid]
val_tags = data['product_group'][valid]

test_posts = data['text'][testid]
test_tags = data['product_group'][testid]




tokenizer = Tokenizer(num_words = vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(train_posts)
word_index = tokenizer.word_index

train_sequences = tokenizer.texts_to_sequences(train_posts)
train_padded = pad_sequences(train_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

val_sequences = tokenizer.texts_to_sequences(val_posts)
val_padded = pad_sequences(val_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

test_sequences = tokenizer.texts_to_sequences(test_posts)
test_padded = pad_sequences(test_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)


encoder = LabelEncoder()
encoder.fit(train_tags)
training_label_seq = encoder.transform(train_tags)
val_label_seq = encoder.transform(val_tags)
test_label_seq = encoder.transform(test_tags)

num_classes = np.max(training_label_seq) + 1


model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(embedding_dim)),    
    tf.keras.layers.Dense(embedding_dim, activation='relu'),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


filepath="weights.best.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]


history = model.fit(train_padded, training_label_seq, 
                    batch_size=batch_size,                   
                    epochs=epochs,
                    verbose=1,
                    callbacks=callbacks_list,
                    validation_data=(val_padded, val_label_seq)) 
                    

model.load_weights("weights.best.hdf5")
#model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

# Evaluate the accuracy of our trained model
score = model.evaluate(test_padded, test_label_seq,batch_size=batch_size, verbose=1)
print('Test score:', score[0])
print('Test accuracy:', score[1])






import itertools
import os

