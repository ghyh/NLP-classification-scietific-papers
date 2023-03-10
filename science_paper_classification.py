# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

import tensorflow as tf
import matplotlib.pyplot as plt
import csv

# global variable
from dataclasses import dataclass

@dataclass
class G:
    validation_ratio = 0.2
    max_feature=4000
    padding_type='pre'
    oov_token='<OOV>'
    data_cols=['TITLE','ABSTRACT']
    AUTOTUNE=tf.data.AUTOTUNE
    embedding_dim=1024
    lstm_dim=16
    epochs=200
    result_dim=4
    batch_size=16
    dropout_ratio=0.2
    patience=2
    np_random_seed=42
    learning_rate = 5e-5

# define the function to read file as csv
def read_input_file(path,filename,delimiter=','):
    filename_with_full_path = os.path.join(path,filename)
    output = None
    with open(filename_with_full_path,'r') as f:
        output = pd.read_csv(f,delimiter=delimiter)
    return output
    
# read train and test file
file_dir = '/kaggle/input/' + 'science-topic-classification/'
train_file = read_input_file(file_dir,'train.csv').drop('ID',axis=1)
test_file = read_input_file(file_dir,'test.csv').drop('ID',axis=1)
print(train_file.info())
train_file.sample(10)

# https://www.tensorflow.org/tutorials/keras/text_classification
def row_standardization(data):
    lower_text = tf.strings.lower(data)
    clean_cells = tf.strings.regex_replace(lower_text,'[^a-zA-Z0-9 ]', ' ')
    clean_cells = tf.strings.regex_replace(clean_cells,'\s{2,}',' ')
    return clean_cells

# visually verify that the non-alphanumeric characters are removed
print(train_file.iloc[4309][0], train_file.iloc[4309][1])  
row_standardization(['ABC%','def'])

# concat all string columns and rows into a big text corpus
def concat_all_string_cells(df,cols_to_concat):
    temp_list = df[cols_to_concat].values.tolist()
    text = " ".join([str(item).lower() for nested_list in temp_list for item in nested_list])
    return text

# unit test for the function
test_df = pd.DataFrame({'col1': ['A','b','c'],'col2': [1,2,3]})
assert concat_all_string_cells(test_df,test_df.columns) == 'a 1 b 2 c 3'

# Randomly select 20% of the train data to be validation data
np.random.seed(G.np_random_seed)

def random_validation_row(df,validation_percentage):
    return np.random.choice(df.index.to_list(),int(len(df.index.to_list())*validation_percentage),False)

# unit testing
unit_test_df = pd.DataFrame({'col1':[1,2,3,4,5,6]})
assert round(len(random_validation_row(unit_test_df,G.validation_ratio))/ len(unit_test_df.index.to_list()),1) == G.validation_ratio

# Split all rows into train and validation sets
# Ref: https://stackoverflow.com/questions/28256761/select-pandas-rows-by-excluding-index-number
excluded_rows = random_validation_row(train_file,G.validation_ratio)
validation_set = train_file.iloc[excluded_rows].reset_index()
train_set = train_file[~train_file.index.isin(excluded_rows)].reset_index()

# cleaning up non-alphanumeric character in train data and obtain parameters for TextVectorization layer
train_corpus = concat_all_string_cells(train_set,G.data_cols)
voc_size = len(train_corpus.split(" "))
clean_train_text = tf.strings.regex_replace(tf.strings.lower(train_set[G.data_cols]),'[^a-zA-Z0-9 ]',' ')
clean_train_text = tf.strings.regex_replace(clean_train_text,'\s{2,}',' ')
max_len = max([len(str(b' '.join(r),'UTF-8')) for r in clean_train_text.numpy().tolist()])
clean_valid_text = tf.strings.regex_replace(tf.strings.lower(validation_set[G.data_cols]),'[^a-zA-Z0-9 ]',' ')
clean_valid_text = tf.strings.regex_replace(clean_valid_text,'\s{2,}',' ')

# set up TexVectorizatin layer
vectorization_layer = tf.keras.layers.TextVectorization(
    #max_tokens=voc_size,
    max_tokens=G.max_feature,
    standardize=row_standardization,
    output_mode='int',
    output_sequence_length=max_len
)

# ref: https://stackoverflow.com/questions/57403472/how-do-i-add-a-new-feature-column-to-a-tf-data-dataset-object
# Generate TF dataset for train and validation set
train_text = np.array([str(b' '.join(r),'UTF-8') for r in clean_train_text.numpy().tolist()])
validation_text = np.array([str(b' '.join(r),'UTF-8') for r in clean_valid_text.numpy().tolist()])
# train dataset 
train_text_ds = tf.data.Dataset.zip((
    tf.data.Dataset.from_tensor_slices(train_text),
    tf.data.Dataset.from_tensor_slices(train_set['label'])
))
train_text_ds = train_text_ds.batch(G.batch_size)
# validation dataset
validation_text_ds = tf.data.Dataset.zip((
    tf.data.Dataset.from_tensor_slices(validation_text),
    tf.data.Dataset.from_tensor_slices(validation_set['label'])
))
validation_text_ds = validation_text_ds.batch(G.batch_size)

vectorization_layer.adapt(tf.data.Dataset.from_tensor_slices(train_text))

# visually check the result of vectorization
vectorization_layer(clean_train_text[0])

# ref: https://towardsdatascience.com/multi-class-text-classification-with-lstm-using-tensorflow-2-0-d88627c10a35
# combine train_text (ipt) and label into a tuple
def data_format(ipt,label):
    ipt = tf.expand_dims(ipt,-1)
    return vectorization_layer(ipt),label

# convert train & validation to Tensorflow dataset
train_input = train_text_ds.map(data_format)
validation_input = validation_text_ds.map(data_format)

# cache and preload data for faster processing
train_input = train_input.cache().prefetch(buffer_size=G.AUTOTUNE)
validation_input = validation_input.cache().prefetch(buffer_size=G.AUTOTUNE)

# ref: https://towardsdatascience.com/multi-class-text-classification-with-lstm-using-tensorflow-2-0-d88627c10a35
# build training model
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(G.max_feature+1,G.embedding_dim),
    tf.keras.layers.Dropout(G.dropout_ratio),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(G.lstm_dim)),
    tf.keras.layers.Dropout(G.dropout_ratio),
    tf.keras.layers.Dense(G.lstm_dim*2,activation='relu'),
    tf.keras.layers.Dropout(G.dropout_ratio),
    tf.keras.layers.Dense(G.lstm_dim,activation='relu'),
    tf.keras.layers.Dropout(G.dropout_ratio),
    tf.keras.layers.Dense(G.result_dim,activation='softmax')
])

model.summary()

# ref: https://angularfixing.com/tensorflow-keras-shape-mismatch/
model.compile(
    loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=tf.optimizers.RMSprop(learning_rate=G.learning_rate),
    metrics=['accuracy']
)

# callback function for training if early stop is needed
callback_early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=G.patience
)

# history of training
history = model.fit(
    train_input,
    validation_data=validation_input,
    callbacks=[callback_early_stop],
    epochs=G.epochs
)

# evaluate the model
history_params = history.history
history_params.keys()

# plot and compare the result of training process
train_accuracy = history_params['accuracy']
train_loss = history_params['loss']
validation_accuracy = history_params['val_accuracy']
validation_loss = history_params['val_loss']

epoch = range(1,len(train_accuracy)+1)
plt.plot(epoch,train_loss,'bo',label='Train Loss')
plt.plot(epoch,validation_loss,'b',label='Validation Loss')
plt.title('Train and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# ref: https://www.tensorflow.org/api_docs/python/tf/keras/Model
# Plot accuracy

plt.plot(epoch, train_accuracy,'bo',label='Train Accuracy')
plt.plot(epoch,validation_accuracy,'b',label='Validation Accuracy')
plt.title('Train and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

## ref: https://www.geeksforgeeks.org/removing-stop-words-nltk-python/
# ref: https://datagy.io/python-list-alphabet/
# remove stop words
from nltk.corpus import stopwords
import string

stop_words = set(stopwords.words('english'))
all_alphabet = list(string.ascii_lowercase)
stop_words = set(list(stop_words) + all_alphabet)

clean_train_text_no_stop_words = clean_train_text
clean_validation_text_no_stop_words = clean_valid_text
# print(clean_train_text_no_stop_words)
for w in stop_words:
    clean_train_text_no_stop_words = tf.strings.regex_replace(clean_train_text_no_stop_words," " + w + " ",' ')    
    clean_validation_text_no_stop_words = tf.strings.regex_replace(clean_validation_text_no_stop_words," " + w + " ",' ')

clean_train_text_no_stop_words = tf.strings.regex_replace(clean_train_text_no_stop_words,"[^a-zA-Z]",' ')
clean_train_text_no_stop_words = tf.strings.regex_replace(clean_train_text_no_stop_words,'\s{2,}',' ')
train_text_no_stop_words = [str(b' '.join(r),'UTF-8') for r in clean_train_text_no_stop_words.numpy().tolist()]
max_len_no_stop_words = max([len(r) for r in train_text_no_stop_words])
train_text_no_stop_words = np.array(train_text_no_stop_words)

# validation set without stop words
clean_validation_text_no_stop_words = tf.strings.regex_replace(clean_validation_text_no_stop_words,"[^a-zA-Z]",' ')
clean_validation_text_no_stop_words = tf.strings.regex_replace(clean_validation_text_no_stop_words,"\s{2,}"," ")
validation_text_no_stop_words = np.array([str(b' '.join(r),'UTF-8') for r in clean_validation_text_no_stop_words.numpy().tolist()])

# re-define a new vectorization layer for corpus that has no stop word
vectorization_layer_no_stop_words = tf.keras.layers.TextVectorization(
    max_tokens=G.max_feature,
    standardize=row_standardization,
    output_mode='int',
    output_sequence_length=max_len_no_stop_words
)
vectorization_layer_no_stop_words.adapt(tf.data.Dataset.from_tensor_slices(train_text_no_stop_words))

# function to associate vectorized text and label
def data_format_no_stop_words(ipt,label):
    ipt = tf.expand_dims(ipt,-1)
    return vectorization_layer_no_stop_words(ipt),label
    
# create new training dataset
train_text_no_stop_words_ds = tf.data.Dataset.zip((
    tf.data.Dataset.from_tensor_slices(train_text_no_stop_words),
    tf.data.Dataset.from_tensor_slices(train_set['label'])
))
train_text_no_stop_words_ds = train_text_no_stop_words_ds.batch(G.batch_size)
# create new validation dataset
validation_text_no_stop_words_ds = tf.data.Dataset.zip((
    tf.data.Dataset.from_tensor_slices(validation_text_no_stop_words),
    tf.data.Dataset.from_tensor_slices(validation_set['label'])
))
validation_text_no_stop_words_ds = validation_text_no_stop_words_ds.batch(G.batch_size)

train_input_no_stop_words = train_text_no_stop_words_ds.map(data_format_no_stop_words)
validation_input_no_stop_words = validation_text_no_stop_words_ds.map(data_format_no_stop_words)

# preload batch data for faster processing
train_input_no_stop_words = train_input_no_stop_words.cache().prefetch(buffer_size=G.AUTOTUNE)
validation_input_no_stop_words = validation_input_no_stop_words.cache().prefetch(buffer_size=G.AUTOTUNE)

# define training model
model_no_stop_words = tf.keras.Sequential([
    tf.keras.layers.Embedding(G.max_feature+1,G.embedding_dim),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(G.lstm_dim)),
    tf.keras.layers.Dropout(G.dropout_ratio),
    tf.keras.layers.Dense(G.lstm_dim*2,activation='relu'),
    tf.keras.layers.Dropout(G.dropout_ratio),
    tf.keras.layers.Dense(G.lstm_dim,activation='relu'),
    tf.keras.layers.Dropout(G.dropout_ratio),
    tf.keras.layers.Dense(G.result_dim,activation='softmax')
])

model_no_stop_words.summary()

# compile model by specifying loss and optimizer
model_no_stop_words.compile(
    loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=tf.optimizers.RMSprop(learning_rate=G.learning_rate),
    metrics=['accuracy']
)

# history of training
history_no_stop_words = model_no_stop_words.fit(
    train_input_no_stop_words,
    validation_data=validation_input_no_stop_words,
    callbacks=[callback_early_stop],
    epochs=G.epochs
)

# ref: https://www.tensorflow.org/tutorials/keras/text_classification
# params of the training process
history_params_no_stop_words = history_no_stop_words.history
train_accuracy_no_stop_words = history_params_no_stop_words['accuracy']
train_loss_no_stop_words = history_params_no_stop_words['loss']
validation_accuracy_no_stop_words = history_params_no_stop_words['val_accuracy']
validation_loss_no_stop_words = history_params_no_stop_words['val_loss']

# plot loss of training & validation
epochs_no_stop_words = range(1,len(train_accuracy_no_stop_words)+1)
plt.plot(epochs_no_stop_words,train_loss_no_stop_words,'bo',label='Train Loss')
plt.plot(epochs_no_stop_words,validation_loss_no_stop_words,'b',label='Validation Loss')
plt.title('Train and Validation Loss without Stop Words')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# plot accuracy of training and validation
plt.plot(epochs_no_stop_words,train_accuracy_no_stop_words,'bo',label='Train Accuracy')
plt.plot(epochs_no_stop_words,validation_accuracy_no_stop_words,'b',label='Validation Accuracy')
plt.title('Train and Validation Accuracy without Stop Words')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# evaluate test data
# test data with stop words and numeric characters
clean_test_text = tf.strings.regex_replace(tf.strings.lower(test_file[G.data_cols]),'[^a-zA-Z0-9 ]',' ')
clean_test_text = tf.strings.regex_replace(clean_test_text,'\s{2,}',' ')
test_text= np.array([str(b' '.join(r),'UTF-8') for r in clean_test_text.numpy().tolist()])
# test dataset
test_text_ds = tf.data.Dataset.from_tensor_slices(test_text)
test_text_ds = test_text_ds.batch(G.batch_size)
test_text_ds = test_text_ds.map(lambda x: vectorization_layer(tf.expand_dims(x,-1)))
test_input = test_text_ds.cache().prefetch(buffer_size=G.AUTOTUNE)

# test data without stop words and numeric characters
clean_test_text_no_stop_words = tf.strings.regex_replace(tf.strings.lower(test_file[G.data_cols]),'[^a-zA-Z0-9 ]',' ')
for w in stop_words:
    clean_test_text_no_stop_words = tf.strings.regex_replace(clean_test_text_no_stop_words, " " + w + " "," ")    
clean_test_text_no_stop_words = tf.strings.regex_replace(clean_test_text_no_stop_words,'[^a-zA-Z]',' ')
clean_test_text_no_stop_words = tf.strings.regex_replace(clean_test_text_no_stop_words,'\s{2,}',' ')
test_text_no_stop_words = np.array([str(b' '.join(r),'UTF-8') for r in clean_test_text_no_stop_words.numpy().tolist()])

test_text_no_stop_words_ds = tf.data.Dataset.from_tensor_slices(test_text_no_stop_words)
test_text_no_stop_words_ds = test_text_no_stop_words_ds.batch(G.batch_size)
test_text_no_stop_words_ds = test_text_no_stop_words_ds.map(lambda x: vectorization_layer_no_stop_words(tf.expand_dims(x,-1)))
test_input_no_stop_words = test_text_no_stop_words_ds.cache().prefetch(buffer_size=G.AUTOTUNE)

# predict resulit for both sets of test data
result = model.predict(test_text_ds,batch_size=G.batch_size)
#print(result)
output = [np.argmax(c) for c in result]
output = np.array([np.arange(1,len(output)+1),output])
print(output)

result_no_stop_words = model_no_stop_words.predict(test_text_no_stop_words_ds,batch_size=G.batch_size)
output_no_stop_words = [np.argmax(c) for c in result_no_stop_words]
output_no_stop_words = np.array([np.arange(1,len(output_no_stop_words)+1),output_no_stop_words])
output_no_stop_words
