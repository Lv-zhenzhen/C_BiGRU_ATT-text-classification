import os

def _read_file(filename):
    """读取一个文件并转换成一行"""
    with open(filename, 'r', encoding='utf-8') as f:
        content = ''.join(f.read().split())
        return content.replace('\n', '').replace('\t', '').replace('\u3000', '')
def merge_file(filedir,f_ouput):
    #filedir = './THUCNews_small/'  # 获取目标文件夹路径
    catelist = os.listdir(filedir)  # 获取corpus_path下的所有子目录
    catelist.sort()
    for mydir in catelist:
        class_path = filedir + mydir + "/"  # 拼出分类子目录的路径如：train_corpus/art/
        #print(class_path)
        file_list = os.listdir(class_path)  # 获取未分词语料库中某一类别中的所有文本
        file_list.sort()
        #file_list.sort(key=lambda x: int(x[:-4]))
        # print(file_list)
        for file_path in file_list:  # 遍历类别目录下的所有文件
            fullname = class_path + file_path  # 拼出文件名全路径如：train_corpus/art/21.txt
            content = _read_file(fullname)
            f_ouput.write(mydir+'\t'+content+'\n')
        print('Finished:', mydir)
if __name__ == '__main__':
    data_path = '../data/THUCNews/'
    merge_content = '../data/merge_file.txt'
    with open(merge_content,'w',encoding='utf-8') as f:
        merge_file(data_path,f)


