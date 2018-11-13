import numpy as np
import gensim
import numpy as np
from sklearn.svm import SVC
from gensim import corpora
from sklearn.externals import joblib
import glob
import os
import time
from sklearn.model_selection import train_test_split
from sklearn import preprocessing, metrics, model_selection
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

# rnn_attention_data = np.loadtxt('./data/word_data/word_dim/word_rnn_attention_embeddings_600_dim8.txt')
char_cnn_data = np.loadtxt('./data/word_data/word_dim/word_rnn_attention_embeddings_600_dim8.txt')
print(char_cnn_data.shape)
y_data = np.loadtxt('./data/doc_cat_id.txt')

np.random.seed(10)
shuffle_indices = np.random.permutation(np.arange(len(y_data)))
x_shuffled = char_cnn_data[shuffle_indices]  # 将文本和标签打乱
y_shuffled = y_data[shuffle_indices]
classfier = SVC(kernel='linear')

accuracy = model_selection.cross_val_score(classfier, x_shuffled,  np.argmax(y_shuffled, axis= 1), cv=10,
                                         scoring='accuracy')
with open('./result/cross_Valid_result.txt', 'a', encoding="utf-8") as f:
    f.write('word_BiGRU_ATT_600_dim8:' + str(accuracy) + "\n")
