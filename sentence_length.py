# -*- coding: utf-8 -*-
# @Time    : 2018/11/20 10:53
# @Author  : QuietWoods
# @FileName: sentence_length.py
# @Software: PyCharm
import re

SENTENCE_MAX_LENGTH = 200  # 句子字符串最大长度


def _merg_seg(text: str, separator="，")->list:
    """
    合并子句，使合并后的子句长度最可能接近100而不超过100。
    exampls:
        "12...3，2...34。" --> ['12...3', '2...34。']
    :param text:
    :return:
    """
    segs = []  # 切割的字符串
    cache_seg = []  # 缓冲列表
    max_len = SENTENCE_MAX_LENGTH  # 限定长度
    nearest_len = 0
    items = text.split(separator)
    sum_seg = len(items)
    for i, item in enumerate(items):
        item_len = len(item)
        if i < sum_seg - 1:
            nearest_len += 1  # 加上省略的逗号
        # 当前子句长度大于限定长度
        if item_len > max_len:
            if cache_seg.__len__():  # 缓冲列表不为空
                segs.append(separator.join(cache_seg))
            segs.append(item)
            nearest_len = 0
            cache_seg = []
        else:
            if nearest_len + item_len > max_len:
                segs.append(separator.join(cache_seg))
                cache_seg = []
                cache_seg.append(item)
                nearest_len = item_len
            elif i < sum_seg - 1:  # 未处理到最后一个item
                nearest_len += item_len
                cache_seg.append(item)
            else:                  # 最后一个item
                cache_seg.append(item)
                segs.append(separator.join(cache_seg))
                cache_seg = []

    # 处理字符串尾部
    if cache_seg:
        segs.append(separator.join(cache_seg))
    # 子句中不存在汉字，过滤。
    segs = filter(lambda seg: re.search('[\u4E00-\u9FFF]', seg), segs)

    return segs


def seg_sent(sentence):
    """
    一个句子的长度大于100，根据分号和引号断句。
    如果没有分号和引号就根据逗号适当的断句，适当的意思是：逗号间隔的子句长度小于100的不超过2句。
    examples:
        tt = "12342"
        seg = tt.split('2')
        ['1', '34', '']
        yy = ".".join(seg)
        '1.34.'
    :param sentence:
    :return:
    """
    if len(sentence) > SENTENCE_MAX_LENGTH:
        if sentence.find('；') >= 0:
            sents = _merg_seg(sentence, '；')
            result = []
            for sent in sents:
                if len(sent) > SENTENCE_MAX_LENGTH:
                    sent = "。".join(_merg_seg(sent, '，'))
                if sent[-1] == '；':
                    sent = sent[:-1] + '。'
                result.append(sent)
            return ''.join(result)
        # elif sentence.find('：') >= 0:
        #     sents = sentence.split('：')
        else:
            sents = _merg_seg(sentence, '，')
        return '。'.join(sents)

    else:
        return sentence



def sentence_duplicated_remove(content):
    """
    句子去重
    :param content:
    :return:
    """
    if not content:
        return None
    unique_sentence = []
    sentences = content.split('。')
    # 不存在句号
    if len(sentences) <= 1:
        return content
    for sent in sentences:
        if sent not in unique_sentence:
            unique_sentence.append(sent)
    return '。'.join(unique_sentence)



