# -*- coding: utf-8 -*-
# @Time    : 2018/7/17 15:16
# @Author  : QuietWoods
# @FileName: make_datafiles.py
# @Software: PyCharm
# @Email    ：1258481281@qq.com
import sys
import os
import collections

import json
import tarfile
import io
import pickle as pkl
import jieba
import re

jieba.load_userdict('mydict.txt')

# acceptable ways to end a sentence
# 根据句号和分号来断句。
END_TOKENS = ['。', '；']

all_train_patents = "patent_number_lists/patent_train.txt"
all_val_patents = "patent_number_lists/patent_test.txt"

patent_tokenized_fulltext_dir = "patent_fulltext_tokenized"
finished_files_dir = "finished_files"

# These are the number of .txt files we expect there to be in patent_fulltext_dir
num_expected_patents_fulltext = 11414


def fixed_bug(content):
    """
    修复说明书中一些句子在下载处理过程中的失误。例如，单独一段的句子没有添加标点符号。
    :param content:
    :return:
    """
    content = re.sub('技术领域', '。技术领域：', content, count=1)
    content = re.sub('背景技术', '背景技术：', content, count=1)
    content = re.sub('发明内容', '发明内容：', content, count=1)
    content = re.sub('具体实施方式', '具体实施方式：', content, count=1)

    # 去除(1),其特征在于，1., 1、，根据权利要求*所述等此类内容。
    content = re.sub('[\(\（]\d+[\)\）]', '', content)
    
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
                    instructions = fixed_bug(data['src_instructions'])
                    # 特殊字段需要格式化
                    special_term = ['src_claim', 'label_abstract', 'src_abstract', 'src_instructions']
                    for k, v in data.items():
                        if k != "src_instructions":
                            if k in special_term:
                                v = format_text(v)
                            data[k] = segments(v)
                        else:
                            # 说明书特殊处理
                            instructions = format_text(instructions)
                            data[k] = segments(instructions)
                    json.dump(data, w_out, ensure_ascii=False, indent=4)


def segments(src_string):
    """
    对字符串分词
    :param src_string: 原始字符串
    :return: 分词后的字符串
    """
    # format text 
    # content = format_text(src_string)
    # 分词
    words = jieba.cut(src_string)
    # 根据句号添加换行符，达到换行的目的。
    split_line = []
    for word in words:
        new_word = fix_missing_period(word)
        split_line.append(new_word)
    return ' '.join(split_line)


def format_text(input_string):
    """
    格式化专利全文：
    1、去掉配方中的量词，把配方中的逗号用顿号替代，使得一个配方成为一句话。（以逗号为基本分句单元）
    2、去掉顿号，用空格替代
    :param input_string: 专利全文中的各部分内容。
    :return: 格式化后的文本内容
    """
    # 匹配表示数量的词，例如："5-30g、", "各300克，", "20%，"；同时会错误的匹配"1、"
    pattern_measure = re.compile('[各]?\d{1,5}[-~.]?\d{0,6}[克份g]?[\d+\%]?[，、]+')
    # 匹配表示数量的词，并且以句号结束。例如："3-8份。"
    pattern_end_period = re.compile('[各]?\d{1,5}[-~]?\d{0,6}[克份g]?[。]+')

    find_all_measures = re.findall(pattern_measure, input_string)
    print('Find all measures: {}'.format(find_all_measures))
    find_end_periods = re.findall(pattern_end_period, input_string)
    print('Find all end period: {}'.format(find_end_periods))

    # 顿号结束的，用空格断开。同时解决错误匹配的序号"1、"
    content = re.sub(pattern=pattern_measure, repl=' ', string=input_string)
    content = re.sub(pattern=pattern_end_period, repl='。', string=content)
    # 用空格替换所有的顿号
    content = re.sub(pattern='、', repl=' ', string=content)
    return content


def tokenize_files_text(map_file):
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
                    content = fin.read()
                    # 解决原始数据的一些缺陷
                    content = fixed_bug(content)
                    words = jieba.cut(content)
                    # 根据句号添加换行符，达到换行的目的。
                    split_line = []
                    for word in words:
                        new_word = fix_missing_period(word)
                        split_line.append(new_word)
                    w_out.write(' '.join(split_line))


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


def read_patent_file(text_file):
    with open(text_file, "r") as f:
        # sentences are separated by 1 newlines
        # single newlines might be image captions
        # so will be incomplete sentence
        lines = f.read().split('\n')
    return lines


def fix_missing_period(word):
    """Adds a period or newline to a line that is missing a period"""
    if word in END_TOKENS:
        return word + '\n'
    else:
        return word


def get_art_abs(patent_file):
    """ return as list of sentences"""
    with open(patent_file, 'r', encoding='utf-8') as fin:
        data = json.load(fin)
    # 解决说明书太长的问题，因为OOM（out of memery)，在这里把说明书句子限定在****
    claim =  data['src_claim'] 
    instructions = data['src_instructions']
    abstract = data['label_abstract']
    # truncated trailing spaces, and normalize spaces
    claim_lines = [line.strip() for line in claim.split('\n')]
    instructions_lines = [line.strip() for line in instructions.split('\n')]
    art_lines = claim_lines[:10] + instructions_lines[:70]
    abst_lines = [line.strip() for line in abstract.split('\n')]
    abst_lines = abst_lines[:7]
    # Separate out article and abstract sentences
    article_lines = []
    abstract_lines = []
    for idx, line in enumerate(art_lines):
        if line == "":
            continue # empty line
        else:
            article_lines.append(line)

    for idx, line in enumerate(abst_lines):
        if line == "":
            continue # empty line
        else:
            abstract_lines.append(line)
    # 解决说明书太长的问题，因为OOM（out of memery)，在这里把说明书句子限定在****
    return article_lines, abstract_lines


def write_to_tar(patent_number_file, out_file, makevocab=False):
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

    if len(sys.argv) != 2:
        print("USAGE: python make_datafiles.py"
              " <patents_corpus_dir>")
        sys.exit()
    patent_fulltext_dir = sys.argv[1]

    # Check the patents directories contain the correct number of .txt files
    check_num_patents(patent_fulltext_dir, num_expected_patents_fulltext)

    # Create some new directories
    if not os.path.exists(patent_tokenized_fulltext_dir):
        os.makedirs(patent_tokenized_fulltext_dir)
    if not os.path.exists(finished_files_dir):
        os.makedirs(finished_files_dir)

    # Run jieba tokenizer on patents dir,
    # outputting to tokenized patents directory
    # tokenize_patents(patent_fulltext_dir, patent_tokenized_fulltext_dir)


    # Read the tokenized patents, do a little postprocessing
    # then write to bin files
    write_to_tar(all_val_patents, os.path.join(finished_files_dir, "val.tar"))
    write_to_tar(all_train_patents, os.path.join(finished_files_dir, "train.tar"),
                 makevocab=True)
