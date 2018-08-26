# -*- coding: utf-8 -*-
# @Time    : 2018/7/17 15:16
# @Author  : QuietWoods
# @FileName: make_datafiles.py
# @Software: PyCharm

import sys
import os
import collections

import json
import tarfile
import io
import pickle as pkl
import jieba
import re

jieba.load_userdict('dict/mydict.txt')

# acceptable ways to end a sentence
END_TOKENS = ['。', '；']

# 语料集划分
all_train_patents = "patent_number_lists/patent_train.txt"
all_val_patents = "patent_number_lists/patent_test.txt"
# 专利全文分词后的目录
patent_tokenized_fulltext_dir = "patent_fulltext_tokenized"
finished_files_dir = "finished_files"

# These are the number of .txt files we expect there to be in patent_fulltext_dir
num_expected_patents_fulltext = 11414


def fixed_instructions_bug(content):
    """
    修复说明书中一些句子在下载处理过程中的失误。例如，单独一段的句子没有添加标点符号。
    :param content:
    :return:
    """
    content = re.sub('技术领域', '。技术领域：', content, count=1)
    content = re.sub('背景技术', '背景技术：', content, count=1)
    content = re.sub('发明内容', '发明内容：', content, count=1)
    content = re.sub('具体实施方式', '具体实施方式：', content, count=1)
    return content


def tokenize_files(map_file):
    """
    根据map.txt对文本内容分词
    :param map_file:
    :return:
    """
    with open(map_file, 'r') as maping:
        for line in maping:
            from_to = line.strip()
            if from_to != "":
                fulltext, tokenize = from_to.split()
                with open(fulltext, 'r', encoding='utf-8') as fin, open(tokenize, 'w', encoding='utf-8') as w_out:
                    # content = fin.read()
                    data = json.load(fin)
                    # 解决说明书的一些断句缺陷
                    data['src_instructions'] = fixed_instructions_bug(data['src_instructions'])
                    for k, v in data.items():
                        # 分词以及断句
                        data[k] = segments(v)
                    json.dump(data, w_out, ensure_ascii=False, indent=4)


def segments(src_string):
    """
    对字符串分词，以及断句
    :param src_string: 原始字符串
    :return: 分词后的字符串
    """
    # 分词
    stop_list = stopword()
    words = jieba.cut(src_string)
    # 根据句号，分号添加换行符，达到换行的目的。
    split_line = []
    for word in words:
        word = word.strip()
        if word != "" and word not in stop_list and not word.isdigit() or word in END_TOKENS:
            new_word = fix_missing_period(word)
            split_line.append(new_word)
    return ' '.join(split_line)


def tokenize_patents(fulltext_dir, tokenized_fulltext_dir):
    """Maps a whole directory of .txt files to a tokenized version using
       jieba Tokenizer
    """
    print("Preparing to tokenize {} to {}...".format(fulltext_dir,
                                                     tokenized_fulltext_dir))
    patents = os.listdir(fulltext_dir)
    # make IO list file
    print("Making list of files to tokenize...")
    with open("mapping.txt", "w") as f:
        for s in patents:
            f.write(
                "{} \t {}\n".format(
                    os.path.join(fulltext_dir, s),
                    os.path.join(tokenized_fulltext_dir, s)
                )
            )
    print("Tokenizing {} files in {} and saving in {}...".format(
        len(patents), fulltext_dir, tokenized_fulltext_dir))
    tokenize_files('mapping.txt')
    print("jieba Tokenizer has finished.")
    os.remove("mapping.txt")

    # Check that the tokenized patents directory contains the same number of
    # files as the original directory
    num_orig = len(os.listdir(fulltext_dir))
    num_tokenized = len(os.listdir(tokenized_fulltext_dir))
    if num_orig != num_tokenized:
        raise Exception(
            "The tokenized patents directory {} contains {} files, but it "
            "should contain the same number as {} (which has {} files). Was"
            " there an error during tokenization?".format(
                tokenized_fulltext_dir, num_tokenized, fulltext_dir, num_orig)
        )
    print("Successfully finished tokenizing {} to {}.\n".format(
        fulltext_dir, tokenized_fulltext_dir))


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


def fix_missing_period(word):
    """Adds a period or newline to a line that is missing a period"""
    if word in END_TOKENS:
        return word + '\n'
    else:
        return word


def get_art_abs(patent_file):
    """
    描述：返回权利要求和说明书，以及摘要。
    return as list of sentences"""
    with open(patent_file, 'r', encoding='utf-8') as fin:
        data = json.load(fin)
    src_claim = data['src_claim']
    src_instructions = data['src_instructions']
    abstract = data['label_abstract']
    # truncated trailing spaces, and normalize spaces
    claim_lines = [line.strip() for line in src_claim.split('\n') if line.strip() != ""]
    instruct_lines = [line.strip() for line in src_instructions.split('\n') if line.strip() != ""]
    abst_lines = [line.strip() for line in abstract.split('\n')]
    # 限制正文的句子最大数量为100，摘要句子数量为10，防止异常样本导致程序出错。
    art_lines = claim_lines[:15] + instruct_lines[:75]
    abst_lines = abst_lines[:10]

    return art_lines, abst_lines


def get_instructions_abs(patent_file):
    """
     描述：返回说明书和摘要。
     return as list of sentences"""
    with open(patent_file, 'r', encoding='utf-8') as fin:
        data = json.load(fin)
    article = data['src_instructions']
    abstract = data['label_abstract']
    # truncated trailing spaces, and normalize spaces
    art_lines = [line.strip() for line in article.split('\n') if line.strip() != ""]
    abst_lines = [line.strip() for line in abstract.split('\n') if line.strip() != ""]
    # 限制正文的句子最大数量为100，摘要句子数量为10，防止异常样本导致程序出错。
    art_lines = art_lines[:100]
    abst_lines = abst_lines[:10]
    return art_lines, abst_lines


def write_to_tar(patent_number_file, out_file, no_claim=True, makevocab=False):
    """Reads the tokenized .txt files corresponding to the urls listed in the
       url_file and writes them to a out_file.
    """
    print("Making bin file for patent_numbers listed in {}...".format(patent_number_file))
    patent_number_list = [line.strip().split('\t')[1] for line in open(patent_number_file)]
    patent_fnames = [s+".json" for s in patent_number_list]
    num_patents = len(patent_fnames)

    if makevocab:
        vocab_counter = collections.Counter()

    with tarfile.open(out_file, 'w') as writer:
        for idx, s in enumerate(patent_fnames):
            if idx % 1000 == 0:
                print("Writing patent {} of {}; {:.2f} percent done".format(
                    idx, num_patents, float(idx)*100.0/float(num_patents)))

            # Look in the tokenized patent dirs to find the .txt file
            # corresponding to this patent_number
            if os.path.isfile(os.path.join(patent_tokenized_fulltext_dir, s)):
                fulltext_file = os.path.join(patent_tokenized_fulltext_dir, s)
            else:
                print("Error: Couldn't find tokenized patent file {} in"
                      " tokenized patent directory {}. Was there an"
                      " error during tokenization?".format(
                          s, patent_tokenized_fulltext_dir))
                # Check again if tokenized patents directory contain correct
                # number of files
                print("Checking that the tokenized patents directory {}"
                      " contains correct number of files...".format(
                          patent_tokenized_fulltext_dir))
                check_num_patents(patent_tokenized_fulltext_dir,
                                  num_expected_patents_fulltext)
                raise Exception(
                    "Tokenized patents directory {}"
                    " contains correct number of files but patent"
                    " file {} found in it.".format(
                        patent_tokenized_fulltext_dir, s)
                )

            # Get the strings to write to .bin file
            if no_claim:
                # 正文中包含说明书
                article_sents, abstract_sents = get_instructions_abs(fulltext_file)
            else:
                # 正文包含说明书和权利要求书
                article_sents, abstract_sents = get_art_abs(fulltext_file)

            # Write to JSON file
            js_example = {}
            js_example['id'] = s.replace('.json', '')
            js_example['article'] = article_sents
            js_example['abstract'] = abstract_sents
            # 添加ensure_ascii=False,可以保证中文在序列化之后不会变成unicode。
            js_serialized = json.dumps(js_example, indent=4, ensure_ascii=False).encode()
            save_file = io.BytesIO(js_serialized)
            tar_info = tarfile.TarInfo('{}/{}.json'.format(
                os.path.basename(out_file).replace('.tar', ''), idx))
            tar_info.size = len(js_serialized)
            writer.addfile(tar_info, save_file)

            # Write the vocab to file, if applicable
            if makevocab:
                art_tokens = ' '.join(article_sents).split()
                abs_tokens = ' '.join(abstract_sents).split()
                tokens = art_tokens + abs_tokens
                tokens = [t.strip() for t in tokens]  # strip
                tokens = [t for t in tokens if t != ""]  # remove empty
                vocab_counter.update(tokens)

    print("Finished writing file {}\n".format(out_file))

    # write vocab to file
    if makevocab:
        print("Writing vocab file...")
        with open(os.path.join(finished_files_dir, "vocab_cnt.pkl"),
                  'wb') as vocab_file:
            pkl.dump(vocab_counter, vocab_file)
        print("Finished writing vocab file")


def check_num_patents(patents_dir, num_expected):
    num_stories = len(os.listdir(patents_dir))
    if num_stories != num_expected:
        raise Exception(
            "patents directory {} contains {} files"
            " but should contain {}".format(
                patents_dir, num_stories, num_expected)
        )


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("USAGE: python make_datafiles.py"
              " <patents_corpus_dir> Yes|No(no_claim)")
        sys.exit()
    patent_fulltext_dir = sys.argv[1]
    no_claim = sys.argv[2]
    if no_claim == "Yes":
        no_claim = True
    else:
        no_claim = False

    # Check the patents directories contain the correct number of .txt files
    check_num_patents(patent_fulltext_dir, num_expected_patents_fulltext)

    # Create some new directories
    if not os.path.exists(patent_tokenized_fulltext_dir):
        os.makedirs(patent_tokenized_fulltext_dir)
    if not os.path.exists(finished_files_dir):
        os.makedirs(finished_files_dir)

    # Run jieba tokenizer on patents dir,
    # outputting to tokenized patents directory
    tokenize_patents(patent_fulltext_dir, patent_tokenized_fulltext_dir)

    # Read the tokenized patents, do a little postprocessing
    # then write to bin files
    write_to_tar(all_val_patents, os.path.join(finished_files_dir, "val.tar"), no_claim=no_claim)
    write_to_tar(all_train_patents, os.path.join(finished_files_dir, "train.tar"), no_claim=no_claim,
                 makevocab=True)
