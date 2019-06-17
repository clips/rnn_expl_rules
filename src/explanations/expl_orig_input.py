from src.utils import get_most_freq_items
from src.explanations.imp_sg import SkipGramVocab

import numpy as np
from nltk import skipgrams


class SeqSkipGram:

    def __init__(self, vocab):
        self.vocab = vocab

    @classmethod
    def from_seqs(cls, seqs, min_n, max_n, skip, topk, vocab, max_vocab_size):

        inst = cls(vocab)

        if inst.vocab is None:
            print("Getting the most frequent skipgrams as vocab")
            inst.init_class(seqs, min_n, max_n, skip, topk, max_vocab_size)

        inst.get_sg_bag(seqs, min_n, max_n, skip)

        return inst

    def init_class(self, seqs, min_n, max_n, skip, topk, max_vocab_size):
        """
        Return tuple containing skip gram sequence terms and their aggregated scores
        :param seqs: list of list of sequences
        :param min_n: minimum ngram length
        :param max_n: maximum ngram length
        :param skip: maximum number of skip positions
        :param topk: number of top skipgrams in every instance to retain
        :param max_vocab_size: largest vocab size
        """

        top_sg_seqs = list()
        for cur_seq in seqs:
            cur_inst_sg_seqs = self.get_sg(cur_seq, min_n, max_n, skip)
            cur_top_sg_seqs = self.get_most_freq_sg(cur_inst_sg_seqs, topk)
            top_sg_seqs.append(cur_top_sg_seqs)

        self.populate_vocab(top_sg_seqs, max_vocab_size=max_vocab_size)

    # @todo: this method is duplicate from imp_sg. Replace to a common method.
    def get_sg(self, seqs, min_n, max_n, skip):
        """
        Return all skipgrams of seqs and scores of length [min_n, max_n] with
        max number of skip tokens=skip
        :param seqs: token sequences 2D list or similar
        :param min_n: minimum skipgram length
        :param max_n: max skipgram length (inclusive)
        :param skip: max number of tokens to skip
        :return: 2D list of skipgrams of seqs and scores (n_inst * n_sg)
        """
        cur_inst_sg_seqs = list()
        for n in range(min_n, max_n + 1):
            if not n:
                continue
            if n == 1:
                cur_inst_sg_seqs.extend(seqs)
                continue

            cur_inst_sg_seqs.extend([' '.join(sg) for sg in skipgrams(seqs, n=n, k=skip)])

        return cur_inst_sg_seqs

    def get_most_freq_sg(self, seqs, k):
        """
        :param sg_seqs: skipgram sequences for a given instance
        :param k: the number of most frequent to retrieve
        :return: set of most frequent k sequences
        """
        freq_sg = get_most_freq_items([seqs], k)
        return freq_sg

    def populate_vocab(self, sg_seqs, max_vocab_size, vocab_filter='freq'):
        self.vocab = SkipGramVocab.create_vocab(seqs=sg_seqs,
                                                max_vocab_size=max_vocab_size,
                                                vocab_filter=vocab_filter)

    def get_sg_bag(self, seqs, min_n, max_n, skip):

        sg_bag = np.zeros(shape=(len(seqs), len(self.vocab)), dtype=np.int8)

        for i, cur_seq in enumerate(seqs):
            cur_inst_sg_seqs = self.get_sg(cur_seq, min_n, max_n, skip)
            for sg in cur_inst_sg_seqs:
                if sg in self.vocab.term2idx:
                    sg_bag[i, self.vocab.term2idx[sg]] = 1

        self.sg_bag = sg_bag
