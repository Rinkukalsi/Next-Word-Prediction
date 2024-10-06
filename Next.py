%pip install tensorflow
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib 
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences

df = pd.read_csv('path_csv')
df.describe()

#Making it into list 

text_list = df1['description'].to_list()

#Tokenization 

tokenizer = Tokenizer()
tokenizer.fit_on_texts(text_list)
total_words = len(tokenizer.word_index)+1   #plus one for counting the space the tokenizer keeps for one extra word

#Text to Sequence
#from lists and mapped words to sequence of lists
tokened = tokenizer.text_to_sequence([text_list]{})

input_seq = [] #vessel

for line in text_list: #each 'line' in the list to split by single sentence to make it easier to feed
    tokened = tokenizer.texts_to_sequences([df][0]) #0 for the zeroth sequence
    for i in range(1, len(tokened)):  #range from 1 because first two words for context
        n_gram_seq = tokened[:i+1]    #slice it at +1 for additional next word
        input_seq.append(n_gram_seq)


merge_seq = []
for seq in input_seq:
    if any(isinstance(element, list) for element in seq) :
         merged = [item for sublist in seq for item in (sublist if isinstance(sublist, list) else [sublist])]
         merge_seq.append(merged)
    else:
        merge_seq.append(seq)


max_seq_len = max([len(sequence)for sequence in merge_seq ])
input_seq_padded = pad_sequences(merge_seq, maxlen = max_seq_len, padding = 'pre') #adding zeros previous to the sentences
input_seq_padded =  np.array(input_seq_padded)

x = input_seq_padded[:,:-1] #everything but last word (column)
y = input_seq_padded[:,-1] # only the last word (column)
y = tf.keras.utils.to_categorical(y, num_classes=total_words) #one hot encoding


model = Sequential()

model.add(Embedding( total_words,100, input_length = max_seq_len-1  ))
model.add(LSTM(150))
model.add(Dense(total_words, activation='softmax' ))


model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
model.fit(x,y , epochs=50, verbose=1)
