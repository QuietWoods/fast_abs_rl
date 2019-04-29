# -*- coding: utf-8 -*-
# @Time    : 2019/4/29 9:42
# @Author  : QuietWoods
# @FileName: PatentAbstractRewrite.py
# @Software: PyCharm
import json
import os
from datetime import timedelta
from time import time
from collections import Counter, defaultdict
from itertools import product
from functools import reduce
import operator as op
from cytoolz import identity, concat, curry
import torch
from torch import multiprocessing as mp

from data.batcher import tokenize

from decoding import Abstractor, RLExtractor, BeamAbstractor
from decoding import make_html_safe
from make_datafiles import tokenize_file


class PatentAbstractRewrite(object):

    def __init__(self):
        self.original_abstract = ''
        self.re_abstract = ''

    def rewrite_abstract(self, original_patent, no_cuda=False):
        self.original_abstract = original_patent
        cuda = torch.cuda.is_available() and not no_cuda
        return self.decode(patent_json=original_patent, cuda=cuda)

    def decode(self, patent_json, model_dir='full_rl_model', beam_size=5,
               diverse=1.0, max_len=60, cuda=False):
        start = time()
        # 设置 model
        with open(os.path.join(model_dir, 'meta.json')) as f:
            meta = json.loads(f.read())
        if meta['net_args']['abstractor'] is None:
            # NOTE: if no abstractor is provided then
            #       the whole model would be extractive summarization
            assert beam_size == 1
            abstractor = identity
        else:
            if beam_size == 1:
                abstractor = Abstractor(os.path.join(model_dir, 'abstractor'),
                                        max_len, cuda)
            else:
                abstractor = BeamAbstractor(os.path.join(model_dir, 'abstractor'),
                                            max_len, cuda)
        extractor = RLExtractor(model_dir, cuda=cuda)
        # 加载专利改写源
        temp_file = 'tmp_tokenized.json'
        # 分词
        print(patent_json)
        print(temp_file)
        print(os.path.exists(patent_json))
        tokenize_file(patent_json, temp_file)
        with open(temp_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            print(data)
            data['article'] = data['src_abstract'] + data['src_instructions']
            raw_article = data['article']
            print(raw_article)

        # prepare save paths and logs
        dec_log = {}
        dec_log['abstractor'] = meta['net_args']['abstractor']
        dec_log['extractor'] = meta['net_args']['extractor']
        dec_log['rl'] = True
        dec_log['patent_json'] = patent_json
        dec_log['beam'] = beam_size
        dec_log['diverse'] = diverse
        print("参数信息：", dec_log)
        # print(raw_article)
        def add_period(sent):
            if sent[-1] != '。':
                sent += ' 。'
            return sent
        # raw_article = [add_period(sent) for sent in raw_article.split('\n')]
        raw_article = [[add_period(sent) for sent in raw_article.split('\n')]]
        # print(raw_article)
        # Decoding
        with torch.no_grad():
            # 对原文分词
            tokenized_article_batch = map(tokenize(None), raw_article)
            ext_arts = []
            ext_inds = []
            for raw_art_sents in tokenized_article_batch:
                ext = extractor(raw_art_sents)[:-1]  # exclude EOE
                # print("ext from extractor:{}\n".format(len(ext)))
                # ext = ext[:10]
                if not ext:
                    # use top-5 if nothing is extracted
                    # in some rare cases rnn-ext does not extract at all
                    ext = list(range(5))[:len(raw_art_sents)]
                else:
                    ext = [i.item() for i in ext]
                ext_inds += [(len(ext_arts), len(ext))]
                ext_arts += [raw_art_sents[i] for i in ext]
            if beam_size > 1:
                all_beams = abstractor(ext_arts, beam_size, diverse)
                dec_outs = rerank_mp(all_beams, ext_inds)
            else:
                dec_outs = abstractor(ext_arts)
            # print(dec_outs)
            decoded_sents = [' '.join(dec) for dec in dec_outs]
            decoded_sents = [add_period(sent) for sent in decoded_sents]
            self.re_abstract = make_html_safe('\n'.join(decoded_sents))
            print('decoded in {} seconds\r'.format(timedelta(seconds=int(time() - start))))
        print(self.re_abstract)
        return self.re_abstract


_PRUNE = defaultdict(
    lambda: 2,
    {1: 5, 2: 5, 3: 5, 4: 5, 5: 5, 6: 4, 7: 3, 8: 3}
)


def rerank_mp(all_beams, ext_inds):
    beam_lists = [all_beams[i: i+n] for i, n in ext_inds if n > 0]
    with mp.Pool(8) as pool:
        reranked = pool.map(rerank_one, beam_lists)
    return list(concat(reranked))


def rerank_one(beams):
    @curry
    def process_beam(beam, n):
        for b in beam[:n]:
            b.gram_cnt = Counter(_make_n_gram(b.sequence))
        return beam[:n]
    beams = map(process_beam(n=_PRUNE[len(beams)]), beams)
    best_hyps = max(product(*beams), key=_compute_score)
    dec_outs = [h.sequence for h in best_hyps]
    return dec_outs


def _make_n_gram(sequence, n=2):
    return (tuple(sequence[i:i+n]) for i in range(len(sequence)-(n-1)))


def _compute_score(hyps):
    all_cnt = reduce(op.iadd, (h.gram_cnt for h in hyps), Counter())
    repeat = sum(c-1 for g, c in all_cnt.items() if c > 1)
    lp = sum(h.logprob for h in hyps) / sum(len(h.sequence) for h in hyps)
    return (-repeat, lp)


if __name__ == '__main__':
    tt = PatentAbstractRewrite()
    tt.rewrite_abstract(r'G:\data\patent\tmp\patent_corpus_deduplicate\CN200710180692.json', False)
