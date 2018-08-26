# -*- coding: utf-8 -*-
# @Time    : 2018/7/28 15:36
# @Author  : QuietWoods
# @FileName: make_my_dict.py
# @Software: PyCharm

import json
import os
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# 设置matplotlib正常显示中文和负号
matplotlib.rcParams['font.sans-serif']=['SimHei']   # 用黑体显示中文
matplotlib.rcParams['axes.unicode_minus']=False     # 正常显示负号


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


def analyze_corpus(patent_fulltext_tokenized):
    """
    分析专利句子长度分布，专利全文句子数量分布。
    :param patent_corpus_dir:
    :return:
    """
    sentence_lengths = []
    claim_lengths = []
    instructions_lengths = []
    label_abstract = []
    count = 0
    files = 0
    for idx, fname in enumerate(os.listdir(patent_fulltext_tokenized)):
        files = idx + 1
        if not fname.endswith('.json'):
            continue
        with open(os.path.join(patent_fulltext_tokenized, fname), 'r', encoding='utf-8') as fin:
            data = json.load(fin)
            claim = _sentences_words(data['src_claim'])
            claim_lengths.append(len(claim))
            sentence_lengths.extend(claim)
            if len(claim) > 100:
                print("{}, claim sentences is:{}".format(fname, len(claim)))
            
            instructions = _sentences_words(data['src_instructions'])
            instructions_lengths.append(len(instructions))
            sentence_lengths.extend(instructions)
            if len(instructions) > 500:
                print("{}, instructions sentences is:{}".format(fname, len(instructions)))
       

            abstract = _sentences_words(data['label_abstract'])
            label_abstract.append(len(abstract))
            if len(abstract) > 20:
                print("{}, label_abstract sentences is:{}".format(fname, len(abstract)))

            if max(claim) > 500 or max(instructions) > 500 or max(abstract) > 500:
                print('{} sentences max length is biger 500'.format(fname))
            count += 1
            if count == 100:
                pass
        print("the {} file.".format(count), end='')
    if count != files:
        print("处理文件数目不一致！处理数目{}，实际数目{}".format(count, files))

    # 画图
    plot_data(sentence_lengths,'words', 'Frequency',  "Patent Sentence Length", (1, 300), 30)
    plot_data(claim_lengths, 'sentence', 'Frequency', "Patent Claim", (1, 40), 40)
    plot_data(instructions_lengths, 'sentence', 'Frequency',  "Patent Instructions", (0, 400), 40)
    plot_data(label_abstract, 'sentence', 'Frequency', "Patent Labeled Abstract", (1, 20), 19)


def plot_data(data, x, y, title, range=None, bins=None):
    """
    绘制直方图
    data:必选参数，绘图数据
    bins:直方图的长条形数目，可选项，默认为10
    normed:是否将得到的直方图向量归一化，可选项，默认为0，代表不归一化，显示频数。normed=1，表示归一化，显示频率。
    facecolor:长条形的颜色
    edgecolor:长条形边框的颜色
    alpha:透明度
    """
    print('*********************{}, {}, {}'.format(title, min(data), max(data)))
    plt.hist(data, bins=bins, range=range,  normed=0, facecolor="blue", edgecolor="black", alpha=0.7)
    # 显示横轴标签
    plt.xlabel(x)
    # 显示纵轴标签
    plt.ylabel(y)
    # 显示图标题
    plt.title(title)
    plt.savefig(title+'.png')
    plt.close()
    # plt.show()


def _sentences_words(input_string):
    """
    统计句子数，每个句子的词数
    :param input_string:
    :return: [words...]
    """
    sents_words = []
    sentences = input_string.split('\n')
    for sent in sentences:
        words = sent.split()
        # 统计句子中词的个数
        sents_words.append(len(words))
    return sents_words


if __name__ == '__main__':
    # make_my_dict('patent_corpus')
    analyze_corpus('patent_fulltext_tokenized')
