import jieba
import jieba.analyse
import xlwt
from utils import data_utils
import numpy as np
import logging
import os

def stop_words(stop_words_file):
    with open(stop_words_file,'r',encoding='utf-8') as f:
        stopwords_list = []
        stopwords_set = set()
        for line in f.readlines():
            stopwords_list.append(line[:-1])
        stopwords_set = set(stopwords_list)
    return stopwords_set
def jieba_fenci(content, output_file):
    #使用结巴分词进行切分,并去停用词
    words = jieba.cut(content, cut_all=False)
    for word in words:
        # if word not in stopwords_set:
        output_file.write(word + ' ')
    output_file.write('\n')

if __name__ == "__main__":
    merge_file = '../data/merge_file.txt'
    # stop_words_file = '../data/word_data/stop_words_ch.txt'
    # stopwords_set = stop_words(stop_words_file)
    output_file = open('../data/word_data/fenci_contents.txt','w',encoding='utf-8')
    labels, contents = data_utils.read_label_content(merge_file)
    for content in contents:
        jieba_fenci(str(content).strip(), output_file)
    output_file.close()