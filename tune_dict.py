# -*- coding: utf-8 -*-
# @Time    : 2018/7/17 15:16
# @Author  : QuietWoods
# @FileName: make_datafiles.py
# @Software: PyCharm

import jieba
import sys
import os
import collections

import json
import tarfile
import io
import pickle as pkl

import re
from collections import OrderedDict

jieba.load_userdict('dict/中药材词典.txt')
jieba.load_userdict('dict/医学术语词典.txt')
# jieba.add_word("制备方法")


def segments(src_string):
    """
    对字符串分词，以及断句
    :param src_string: 原始字符串
    :return: 分词后的字符串
    """
    # 分词
    src_string = "肝病；肝炎；更年期综合征；痴呆；哮喘；脑梗死；护肤；肾病；心脏病；抑郁症；炎性肠病；腹泻；炎症；阿尔茨海默病；麻痹；视力减退；便秘；抗真菌；利尿剂；干皮病；皮炎；毛发生长促进剂；色素沉着；脑卒中后遗症；强肌肉剂；机能性肠病；机能性心脏病；机能性肝脏病；机能性肾脏病；皮真菌病；发汗；皮肤损伤；特应性皮炎；止汗剂；运动麻痹；疣；老年性干皮症；老年性角质瘤；消化液分泌促进剂；风湿性关节炎；关节痛；风湿；关节炎；胃溃疡；头痛；腰痛；眩晕；运动障碍；甲状腺功能减退症；克罗恩病；变应性大肠炎；耳鸣；肺癌；腹痛；健忘症；疲劳；痔；淋巴瘤；肿瘤；癌；内出血；乳腺癌；黏膜白斑；直肠膨出；贫血；虚弱；脑瘤；脱发；手足痛；长期疲劳；原子弹爆炸"
    print(src_string)
    stop_list = stopword()
    words = jieba.cut(src_string, HMM=False)

    # 根据句号，分号添加换行符，达到换行的目的。
    split_line = []
    for word in words:
        if word.strip() != "":
            if word not in stop_list and not word.isdigit():
                split_line.append(word)
    print(split_line)
    return ' '.join(split_line)

def main():
    with open("G:\\data\\patent\\tmp\\patent_corpus\\CN200610060654.json", 'r', encoding="utf-8") as f:
        data = json.loads(f.read(), object_pairs_hook=OrderedDict)
        for k, v in data.items():
            data[k] = segments(v)

    with open("C:\\Users\\wl\\Downloads\\CN200610038743_seg.json", "w", encoding='utf-8') as w:
        json.dump(data, w, ensure_ascii=False, indent=4)


def stopword():
    """
    加载停用词表
    :return: set
    """
    stopword_set = set()
    with open('dict/stopword.txt', 'r', encoding='utf-8') as f:
        for line in f:
            word = line.strip()
            if word != "":
                stopword_set.add(word)
    return stopword_set


def make_map_files(patent_number_file, out_file):
    """
    """
    print("Making bin file for patent_numbers listed in {}...".format(patent_number_file))
    patent_number_list = [line.strip().split('\t')[1] for line in open(patent_number_file)]
    patent_fnames = [s+".json" for s in patent_number_list]
    with open(out_file, 'w') as w:
        for idx, s in enumerate(patent_fnames):
            w.write("{}\t{}\n".format(idx, s))

        print("Finished writing file {}\n".format(out_file))

# 语料集划分
all_train_patents = "patent_number_lists/patent_train.txt"
all_val_patents = "patent_number_lists/patent_test.txt"
# 专利全文分词后的目录
patent_tokenized_fulltext_dir = "patent_fulltext_tokenized"
finished_files_dir = "finished_files"

if __name__ == '__main__':
    main()
    # if len(sys.argv) != 3:
    #     print("USAGE: python make_datafiles.py"
    #           " <patents_corpus_dir> Yes|No(no_claim)")
    #     sys.exit()
    # patent_fulltext_dir = sys.argv[1]
    # no_claim = sys.argv[2]
    # if no_claim == "Yes":
    #     no_claim = True
    # else:
    #     no_claim = False
    #
    # make_map_files(all_val_patents, os.path.join(finished_files_dir, "val_map.txt"))
    # make_map_files(all_train_patents, os.path.join(finished_files_dir, "train_map.txt"))
