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
# print(rnn_attention_data.shape)
# concat_cnn_BiGRU_ATT = np.concatenate((rnn_attention_data,cnn_data), axis=1)
# print(concat_cnn_BiGRU_ATT.shape)
# np.savetxt("./concat_cnn_20_30_dim8_BIGRU_600_dim8.txt", concat_cnn_BiGRU_ATT)
y_data = np.loadtxt('./data/doc_cat_id.txt')

np.random.seed(10)
shuffle_indices = np.random.permutation(np.arange(len(y_data)))
x_shuffled = char_cnn_data[shuffle_indices]  # 将文本和标签打乱
y_shuffled = y_data[shuffle_indices]

# dev_sample_index = -1 * int(0.1 * float(len(y_data)))  # 打乱后前90%为训练数据，后10%为测试数据
# x_train, x_test = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
# y_train, y_test = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]
# print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_test)))

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
with open('./result/word_BiGRU_ATT_600_dim8.txt', 'a', encoding="utf-8") as f:
    f.write(str(result) + "\n")



# x_train, x_test, y_train, y_test = train_test_split(concat_rnn_cnn_data, y_data, test_size=0.1, stratify=y_data, random_state=0)
# print(x_test.shape)
# print(y_test.shape)
# print(len(x_train))
# clf = SVC(kernel='linear')
# # clf = SVC(kernel='linear',C=7.806507,gamma=0.000001)
# print("Training a SVM Classifier.")
# clf=clf.fit(x_train, np.argmax(y_train, axis= 1))
# pred = clf.predict(x_test)
# # print(pred,y_dev)
# print(pred)
# accuracy=np.mean(pred == np.argmax(y_test, axis= 1))
# t1 = time.time()
# print('The classification accuracy is %f' % accuracy)
# print("Precision, Recall and F1-Score:\n\n",
#                   classification_report(np.argmax(y_test, axis= 1), pred, target_names=['体育', '军事', '医学', '文化', '汽车', '经济']))