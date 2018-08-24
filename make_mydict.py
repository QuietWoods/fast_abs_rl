# -*- coding: utf-8 -*-
# @Time    : 2018/7/28 15:36
# @Author  : QuietWoods
# @FileName: make_my_dict.py
# @Software: PyCharm
# @Email    ：1258481281@qq.com
import json
import os
import re


def make_my_dict(corpus_dir):
    """
    通过标注语料的label_IT创建自定义词表
    :param corpus_dir:
    :return:
    """
    print('通过标注语料的label_IT创建自定义词表...')
    mydict = {}
    for idx, fnames in enumerate(os.listdir(corpus_dir)):
        with open(os.path.join(corpus_dir, fnames), 'r', encoding='utf-8') as fin:
            data = json.load(fin)
            try:
                label_IT = data['label_IT']
                terms = label_IT.split('；')
                for term in terms:
                    if re.search('[a-zA-Z]', term) is not None:
                        # 如果存在英文，结束当前循环
                        break
                    if term in mydict:
                        mydict[term] += 1
                    else:
                        mydict[term] = 1
            except KeyError as e:
                print(e)
        if idx % 1000 == 0:
            print('deal with :{}'.format(idx))

    with open('mydict.txt', 'w', encoding='utf-8') as fout:
        for k, v in mydict.items():
            fout.write('{} {} n\n'.format(k, v))
    print('通过标注语料的label_IT创建自定义词表，完成！')


if __name__ == '__main__':
    make_my_dict('patent_corpus')

