
from gensim import corpora,models
from utils import data_utils
import tensorflow.contrib.keras as kr
import os
from gensim.models import  word2vec

def dct(f_fenci):
    #用于保存分词后的词典
    fenci_contents = []
    with open(f_fenci, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            fenci_contents.append(line)
            #print(line.strip())
    dct = corpora.Dictionary(line.lower().split() for line in fenci_contents)
    return dct
if __name__ == "__main__":
    f_fenci = '../data/word_data/fenci_contents.txt'
    '''将分词后的文件保存为词典'''
    f_dict = '../data/word_data/fenci_contents_dct.dict'
    # f_dict_txt = '../data/word_data/words_vocab_to_id.txt'
    f_dict_txt = '../data/word_data/words_vocab.txt'
    dct = dct(f_fenci)
    dct.save(f_dict)
    print('dictionary.token2id', dct.token2id)#{'10': 0, '25': 1, '27': 2, '29': 3, '31': 4, '一位': 5,
    with open(f_dict_txt,'w',encoding='utf-8') as f:
        # f.write('pad' + '\n')
        f.write('pad' + '\t' + '0' + '\n')
        for (k, v) in dct.items():
            f.write(str(v)+ '\n')
            # f.write(str(v) + '\t' + str(k+1) + '\n')