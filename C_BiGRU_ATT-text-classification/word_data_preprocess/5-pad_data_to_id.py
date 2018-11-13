import tensorflow.contrib.keras as kr
import numpy as np

def open_file(filename):
    '''打开文件'''
    return open(filename,'r',encoding='utf-8', errors='ignore')
def native_content(content):
    return content
def read_vocab(vocab_dir):
    """读取字符级词汇表"""
    # words = open_file(vocab_dir).read().strip().split('\n')
    with open_file(vocab_dir) as fp:
        words = [native_content(_.strip()) for _ in fp.readlines()]
    word_to_id = dict(zip(words, range(len(words))))
    return words, word_to_id
def data_to_id(filename, word_to_id):
    '''将文件转换为id表示'''

    data_id = []
    with open(filename, 'r', encoding='utf-8') as f:
        fenci_contents_list = []
        for line in f.readlines():
            fenci_contents_list.append(line.split())
    for i in range(len(fenci_contents_list)):
         data_id.append([word_to_id[x] for x in fenci_contents_list[i] if x in word_to_id])
    return data_id
def to_words(content, words):
    """将id表示的内容转换成文字"""
    return ''.join(words[x] for x in content)
'''将id表示的文本内容pad到max_length长度'''
def pad_sentence(data_id, max_length):
    x_pad = kr.preprocessing.sequence.pad_sequences(data_id, max_length, padding='post', truncating='post')
    return x_pad
if __name__ == '__main__':
    vocab_path = '../data/word_data/words_vocab.txt'
    filename = '../data/word_data/fenci_contents.txt'
    words, word_to_id = read_vocab(vocab_path)
    '''将词语转换成id表示'''
    data_id = data_to_id(filename,word_to_id)
    '''将id表示的内容转换成文字输出'''
    for line in data_id[0:1]:
       print(to_words(line,words))
    '''将pad后的data_id保存到文件中'''
    for max_length in [100,200,300,400,500,600,700,800] :
        x_pad = pad_sentence(data_id, max_length)
        x_pad_arr = np.array(x_pad)
        np.savetxt('../data/word_data/pad_data_to_id/pad_data_to_id_' + str(max_length) + '.txt', x_pad_arr,fmt='%d')