
import os
import numpy as np

def native_content(content):
    return content
def open_file(filename):
    '''打开文件'''
    return open(filename,'r',encoding='utf-8', errors='ignore')
def read_label_content(filename):
    '''读取文件内容和标签'''
    contents, labels = [], []
    with open_file(filename) as f:
        lines = f.readlines()
        for line in lines:
            label_content = line.split('\t')
            labels.append(label_content[0])
            contents.append(label_content[1])
    return labels, contents
def read_vocab(vocab_dir):
    """读取字符级词汇表"""
    # words = open_file(vocab_dir).read().strip().split('\n')
    with open_file(vocab_dir) as fp:
        words = [native_content(_.strip()) for _ in fp.readlines()]
    word_to_id = dict(zip(words, range(len(words))))
    return words, word_to_id
def read_category(categories):
    """读取分类目录¨"""
    #categories = ['体育', '娱乐', '家居', '彩票', '彩票', '房产', '教育', '时尚', '时政', '星座','游戏']
    categories = [native_content(x) for x in categories]
    cat_to_id = dict(zip(categories, range(len(categories))))
    return categories, cat_to_id
def load_word2id(path):
    """
    :param path: word_to_id词汇表路径
    :return: word_to_id:{word: id}
    """
    word_to_id = {}
    with open(path, encoding='utf-8') as f:
        for line in f.readlines():
            sp = line.strip().split()
            word = sp[0]
            idx = int(sp[1])
            if word not in word_to_id:
                word_to_id[word] = idx
    return word_to_id

def batch_iter(data, batch_size, shuffle=False):
    """
    Generate a batch iterator for dataset
    """
    data_size = len(data)
    data = np.array(data)
    num_batches_per_epoch = int((data_size - 1) / batch_size) + 1
    # Shuflle the data in training
    if shuffle:
        indices_shuffled = np.random.permutation(np.arange(data_size))
        data_shuffled = data[indices_shuffled]
    else:
        data_shuffled = data
    for batch_num in range(num_batches_per_epoch):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        yield data_shuffled[start_index:end_index]
def real_len(x_batch):
    """
    Get actual lengths of sequences
    """
    return np.array([len(x) for x in x_batch])