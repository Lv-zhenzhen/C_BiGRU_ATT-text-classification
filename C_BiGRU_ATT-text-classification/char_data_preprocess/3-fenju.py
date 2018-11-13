# -*-coding=UTF-8-*-
from utils import data_utils
import re
from collections import Counter
import numpy as np

def fenju(merge_file):#文件进行分句，返回['一方面是公认的中国足球市场日渐萧条', '一方面是共有540人荣登转会上榜名单'...]
    _, contents = data_utils.read_label_content(merge_file)
    all_data_sent = []
    for doc in contents :
        all_data_sent_temp = []
        for sent in re.split(u'，|。|、|；|……|？|！|——|:', doc):
            all_data_sent_temp.append(sent)
        all_data_sent.append(all_data_sent_temp)
    # print(all_data_sent)
    return all_data_sent
def min_char_in_sent(all_data_sent,max_char_in_sent = 4):#只读取句子中字符个数大于max_char_in_sent
    all_data_max_sent = []
    for i,doc in enumerate(all_data_sent):
        all_data_max_sent_temp = []
        for j,sent in enumerate(doc):
            c = Counter(sent)
            if len(c) > max_char_in_sent:
                # print(sent)
                all_data_max_sent_temp.append(sent)
        all_data_max_sent.append(all_data_max_sent_temp)
    return all_data_max_sent
#将文章转换成30*30的矩阵,用id表示
PAD = 0
def all_data_to_id(vocab_file,all_data_max_sent ,num_docs, max_sent_in_doc = 30, max_char_in_sent = 20):
    _, word_to_index = data_utils.read_vocab(vocab_file)#读取字典{'<PAD>': 0, '，': 1, '的': 2, '。': 3,...}
    doc_to_id = np.zeros([num_docs, max_sent_in_doc, max_char_in_sent], dtype=int)
    for doc_index, doc in enumerate(all_data_max_sent):
        sent_to_id = np.zeros([max_sent_in_doc, max_char_in_sent])
        for sent_index,sent in enumerate(doc):
            if sent_index < max_sent_in_doc :
                word_to_id = np.zeros([max_char_in_sent], dtype=int)
                for char_index, char in enumerate(Counter(sent)):
                    # print(char)
                    if char_index < max_char_in_sent:
                        word_to_id[char_index] = word_to_index.get(char, PAD)
                        # print(word_to_id)
                sent_to_id[sent_index] = word_to_id
        doc_to_id[doc_index] = sent_to_id
    return doc_to_id
if __name__ == '__main__':
    merge_file = '../data/merge_file.txt'
    all_data_sent = fenju(merge_file)
    all_data_max_sent = min_char_in_sent(all_data_sent,max_char_in_sent = 4)
    print(all_data_max_sent)
    vocab_file = vocab_file = '../data/char_data/char_vocab.txt'
    doc_to_id = all_data_to_id(vocab_file, all_data_max_sent, 20480, 30, 10)
    # print(doc_to_id)
    '''将id形成的文章表示写入文件中'''
    doc_char_to_id = '../data/char_data/pad_data_to_id/doc_char_to_id_30_10.txt'
    with open(doc_char_to_id, 'w',encoding='utf-8') as f:
        f.write('#Array shape:{0}\n'.format(doc_to_id.shape))
        for data_slice in doc_to_id:
            np.savetxt(f, data_slice, fmt='%d')
            f.write('#New slice\n')
    '''将id表示的文章加载'''
    # new_data = np.loadtxt(doc_char_to_id)
    # new_data = new_data.reshape(doc_to_id.shape)
    # print(new_data.shape)
    # print(new_data)