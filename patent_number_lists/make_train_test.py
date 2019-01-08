# -*- coding: utf-8 -*-
# @Time    : 2018/7/18 20:06
# @Author  : QuietWoods
# @FileName: make_train_test.py
# @Software: PyCharm

import os
from os.path import join
from sklearn.model_selection import train_test_split


def make_train_test(patent_number_lists):
    """
    切割数据集
    :param patent_number_lists:
    :return:
    """
    patent_list = []
    with open(patent_number_lists, 'r') as fin:
        for line in fin:
            patent_number = line.strip()
            if patent_number != "":
                patent_list.append(patent_number)

    train, test = train_test_split(patent_list, test_size=0.2)

    with open('patent_train.txt', 'w', encoding='utf-8') as train_out, \
            open('patent_test.txt', 'w', encoding='utf-8') as test_out:
        train_out.write('\n'.join(train))
        test_out.write('\n'.join(test))


def prepare_for_data(fulltext, abstract):
    """
    从下载的专利全文中提取权利要求和说明书，从标注的语料中提取摘要。
    :param fulltext:
    :param abstract:
    :return:
    """
    patent_fulltext_dir = "G:\\data\\patent\\text"
    patent_abstract_dir = "G:\\data\\patent\\abstract"
    if not os.path.exists(patent_abstract_dir):
        os.mkdir(patent_abstract_dir)
    if not os.path.exists(patent_fulltext_dir):
        os.mkdir(patent_fulltext_dir)
    patents = os.listdir(abstract)
    for idx, fname in enumerate(patents):
        with open(join(abstract, fname), 'r', encoding='utf-8') as abs_in, \
                open(join(fulltext, fname), 'r', encoding='utf-8') as fulltext_in:
            with open(join(patent_fulltext_dir, fname), 'w', encoding='utf-8') as pat_text, \
                    open(join(patent_abstract_dir, fname), 'w', encoding='utf-8') as pat_abs:
                # deal with fulltext
                title, abst, claim, instructions = fulltext_in.read().strip().split('\n')
                # 权利要求和说明书
                pat_text.write(claim + '\n' + instructions)
                # deal with abstract
                human_abst = abs_in.read().split('\n')
                # 人工改写的摘要
                pat_abs.write(human_abst[1])
        if idx % 1000 == 0:
            print(idx)
    print("prepare data finished.")


if __name__ == '__main__':
    # fulltext = "G:\\data\\patent\\tmp\\text"
    # abstract = "G:\\data\\patent\\tmp\\abstract"
    # prepare_for_data(fulltext=fulltext, abstract=abstract)
    with open('patent_number_lists.txt', 'w', encoding='utf-8') as w:
        for idx, fname in enumerate(os.listdir(r"G:\data\patent\tmp\filter_patents")):
            w.write('{}\t{}\n'.format(idx + 1, fname.split('.')[0]))
    make_train_test('patent_number_lists.txt')
