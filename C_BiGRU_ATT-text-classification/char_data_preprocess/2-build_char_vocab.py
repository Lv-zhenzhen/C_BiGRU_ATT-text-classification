from utils import data_utils
from collections import Counter
import numpy as np
import tensorflow.contrib.keras as kr
import os

def build_vocab(merge_file, vocab_file, vocab_size):
    """根据字符级数据集构建词汇表，存储"""
    _, contents = data_utils.read_label_content(merge_file)
    all_data = []
    for content in contents:
        all_data.extend(content)
    counter = Counter(all_data)#{'，': 480926, '的': 348828, '。': 194675, '一': 119858}
    count_pairs = counter.most_common(vocab_size - 1)#[('a', 5), ('b', 4), ('c', 3)]
    print(vocab_size-1)
    words, _ = list(zip(*count_pairs))
    print(len(list(words)))
    # 添加一个<pad>将所有文本pad为同一个长度
    words = ['PAD'] + list(words)
    open(vocab_file,'w',encoding='utf-8').write('\n'.join(words) + '\n')
if __name__ == '__main__':
    merge_file = '../data/merge_file.txt'
    vocab_file = '../data/char_data/char_vocab.txt'
    build_vocab(merge_file, vocab_file, 5000)
    words,word_to_id = data_utils.read_vocab(vocab_file)
    print(word_to_id)