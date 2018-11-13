from utils import data_utils
import tensorflow.contrib.keras as kr
import numpy as np

#将标签转换成 id表示
def categories_to_id(categories):
    categories, cat_to_id = data_utils.read_category(categories)
    return categories, cat_to_id
def categories_to_hot(merge_file, cat_to_id):
    labels, _ = data_utils.read_label_content(merge_file)
    label_id = []
    for i in range(len(labels)):
        label_id.append(cat_to_id[labels[i]])
    cat_pad = kr.utils.to_categorical(label_id, num_classes=len(cat_to_id))
    return cat_pad

if __name__ == '__main__':
    merge_file = './data/merge_file.txt'
    doc_to_id_file = './data/doc_cat_id.txt'
    categories = ['体育', '军事','医学','文化','汽车','经济']
    _, cat_to_id = categories_to_id(categories)
    # cat_pad = categories_to_hot(merge_file, cat_to_id)
    # np.savetxt(doc_to_id_file, cat_pad, fmt='%d')
    # print(cat_pad)
    # print(len(cat_pad))
    # print(cat_pad.shape)
    '''加载txt文件'''
    new_data = np.loadtxt(doc_to_id_file)
    print(len(new_data[1]))


