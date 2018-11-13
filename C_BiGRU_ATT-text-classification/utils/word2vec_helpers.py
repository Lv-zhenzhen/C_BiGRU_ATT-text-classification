import os
import sys
import logging
import multiprocessing
import time
import json
import gensim
import numpy as np

from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
from gensim.models import word2vec

def output_vocab(vocab):
    for k, v in vocab.items():
        print(k)

def embedding_sentences(sentences, embedding_size=128, window=5, min_count=1, file_to_load=None, file_to_save=None):
    if file_to_load is not None:
        w2vModel = Word2Vec.load(file_to_load)
    else:
        w2vModel = Word2Vec(sentences, size=embedding_size, window=window, min_count=min_count,
                            workers=multiprocessing.cpu_count())
        if file_to_save is not None:
            w2vModel.save(file_to_save)
    all_vectors = []
    embeddingDim = w2vModel.vector_size
    embeddingUnknown = [0 for i in range(embeddingDim)]
    for sentence in sentences:
        this_vector = []
        for word in sentence:
            if word in w2vModel.wv.vocab:
                this_vector.append(w2vModel[word])
            else:
                this_vector.append(embeddingUnknown)
        all_vectors.append(this_vector)
    return all_vectors


def generate_word2vec_files(input_file, output_model_file, output_vector_file, size=128, window=5, min_count=1):
    start_time = time.time()

    # trim unneeded model memory = use(much) less RAM
    # model.init_sims(replace=True)
    model = Word2Vec(LineSentence(input_file), size=size, window=window, min_count=min_count,
                     workers=multiprocessing.cpu_count())
    model.save(output_model_file)
    model.wv.save_word2vec_format(output_vector_file, binary=False)

    end_time = time.time()
    print("used time : %d s" % (end_time - start_time))

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


def build_word2vec(fname, word2id, save_to_path=None):
    """
    :param fname: 预训练的word2vec.
    :param word2id: 语料文本中包含的词汇集.
    :param save_to_path: 保存训练语料库中的词组对应的word2vec到本地
    :return: 语料文本中词汇集对应的word2vec向量{id: word2vec}.
    """
    n_words = max(word2id.values()) + 1
    model = word2vec.Word2Vec.load(fname)
    word_vecs = np.array(np.random.uniform(-1., 1., [n_words, model.vector_size]))
    for word in word2id.keys():
        try:
            word_vecs[word2id[word]] = model[word]
        except KeyError:
            pass
    if save_to_path:
        with open(save_to_path, 'w', encoding='utf-8') as f:
            for vec in word_vecs:
                vec = [str(w) for w in vec]
                f.write(' '.join(vec))
                f.write('\n')
    return word_vecs


# word2id = load_word2id('../data/words_result/words_vocab.txt')
# w2v = build_word2vec('../data/word2vec_result/w2v_128.model', word2id, save_to_path='../data/word2vec_result/corpus_word2vec_128.txt')

def load_corpus_word2vec(path):
    """加载语料库word2vec词向量,相对wiki词向量相对较小"""
    word2vec = []
    with open(path, encoding='utf-8') as f:
        for line in f.readlines():
            sp = [float(w) for w in line.strip().split()]
            word2vec.append(sp)
    return np.asarray(word2vec)
