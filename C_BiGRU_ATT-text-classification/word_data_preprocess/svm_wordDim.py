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

filedir = '../data/word_data/word_dim/'  # 获取目标文件夹路径
file_list = os.listdir(filedir)  # 获取未分词语料库中某一类别中的所有文本
for file_path in file_list:  # 遍历类别目录下的所有文件
    fullname = filedir + file_path  # 拼出文件名全路径如：train_corpus/art/21.txt
    # print(fullname)
    cnn_data = np.loadtxt(fullname)
    y_data = np.loadtxt('../data/doc_cat_id.txt')

    np.random.seed(10)
    shuffle_indices = np.random.permutation(np.arange(len(y_data)))
    x_shuffled = cnn_data[shuffle_indices]  # 将文本和标签打乱
    y_shuffled = y_data[shuffle_indices]
    classfier = SVC(kernel='linear')
# cross_validation.cross_val_score
    f1_macro = model_selection.cross_val_score(classfier, x_shuffled,  np.argmax(y_shuffled, axis= 1), cv=10,
                                           scoring='f1_macro').mean()
    f1_micro = model_selection.cross_val_score(classfier, x_shuffled,  np.argmax(y_shuffled, axis= 1), cv=10,
                                           scoring='f1_micro').mean()
    accuracy = model_selection.cross_val_score(classfier, x_shuffled,  np.argmax(y_shuffled, axis= 1), cv=10,
                                           scoring='accuracy').mean()
    precision_macro = model_selection.cross_val_score(classfier, x_shuffled,  np.argmax(y_shuffled, axis= 1), cv=10,
                                                  scoring='precision_macro').mean()
    precision_micro = model_selection.cross_val_score(classfier, x_shuffled,  np.argmax(y_shuffled, axis= 1), cv=10,
                                                  scoring='precision_micro').mean()
    recall_macro = model_selection.cross_val_score(classfier, x_shuffled,  np.argmax(y_shuffled, axis= 1), cv=10,
                                               scoring='recall_macro').mean()
    recall_micro = model_selection.cross_val_score(classfier, x_shuffled,  np.argmax(y_shuffled, axis= 1), cv=10,
                                               scoring='recall_micro').mean()
    # roc_auc = model_selection.cross_val_score(classfier, model_matrix_shuffled, model_label_shuffled, cv=10, scoring='roc_auc').mean()

    result = ('f1_macro:', f1_macro, 'f1_micro(accuracy):', f1_micro, 'accuracy:', accuracy,
          'precision_macro:', precision_macro, 'precision_micro:', precision_micro,
          'recall_macro:', recall_macro, 'recall_micro:', recall_micro,)
    with open('../result/svm_wordDim/wordDim.txt', 'a', encoding="utf-8") as f:
        f.write(fullname + ':' + str(result) + "\n")