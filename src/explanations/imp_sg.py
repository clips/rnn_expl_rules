from src.utils import get_top_items_dict

from nltk import skipgrams
import numpy as np
from operator import itemgetter
from collections import Counter, defaultdict

class SeqSkipGram:

    def __init__(self, vocab, pos_th, neg_th):
        self.vocab = vocab
        self.pos_th = pos_th
        self.neg_th = neg_th

    @classmethod
    def from_seqs(cls, seqs, scores, min_n, max_n, skip, topk, vocab, max_vocab_size, pos_th, neg_th):

        inst = cls(vocab, pos_th, neg_th)

        if inst.vocab is None:
            print("Getting most important skipgrams as vocab")
            inst.init_class(seqs, scores, min_n, max_n, skip, topk, max_vocab_size)

        inst.get_sg_bag(seqs, scores, min_n, max_n, skip)

        print("pos neg th", inst.pos_th, inst.neg_th)

        return inst

    def init_class(self, seqs, scores, min_n, max_n, skip, topk, max_vocab_size):
        """
        Return tuple containing skip gram sequence terms and their aggregated scores
        :param seqs: list of list of sequences
        :param scores: list of list of importance scores of sequences
        :param min_n: minimum ngram length
        :param max_n: maximum ngram length
        :param skip: maximum number of skip positions
        :param topk: number of top skipgrams in every instance to retain
        """

        top_sg_seqs,  top_sg_scores = list(), list()
        for cur_seq, cur_score in zip(seqs, scores):
            cur_inst_sg_seqs, cur_inst_sg_scores = self.get_sg(cur_seq, cur_score, min_n, max_n, skip)
            cur_top_sg_seqs, cur_top_sg_scores = self.get_top_sg(cur_inst_sg_seqs, cur_inst_sg_scores, topk)
            top_sg_seqs.append(cur_top_sg_seqs)
            top_sg_scores.append(cur_top_sg_scores)

        self.populate_vocab(top_sg_seqs, top_sg_scores, max_vocab_size)

        # use minimum imp of top skipgrams as threshold.
        self.pos_th = min([j for i in top_sg_scores for j in i if j > 0.])
        self.neg_th = min([j for i in top_sg_scores for j in i if j < 0.])

    def get_sg(self, seqs, scores, min_n, max_n, skip):
        """
        Return all skipgrams of seqs and scores of length [min_n, max_n] with max number of skip tokens/=
        :param seqs: token sequences 2D list or similar
        :param scores: imp score sequences 2D list or similar
        :param min_n: minimum skipgram length
        :param max_n: max skipgram length (inclusive)
        :param skip: max number of tokens to skip
        :return: 2D list of skipgrams of seqs and scores (n_inst * n_sg)
        """
        cur_inst_sg_seqs, cur_inst_sg_scores = list(), list()
        for n in range(min_n, max_n + 1):
            if not n:
                continue
            if n == 1:
                cur_inst_sg_seqs.extend(seqs)
                cur_inst_sg_scores.extend(scores)
                continue

            cur_inst_sg_seqs.extend([' '.join(sg) for sg in skipgrams(seqs, n=n, k=skip)])
            cur_inst_sg_scores.extend([np.mean(sg) for sg in skipgrams(scores, n=n, k=skip)])

        return cur_inst_sg_seqs, cur_inst_sg_scores

    def get_top_sg(self, seqs, scores, k):
        """
        Get the top k skipgrams for a given instance and their corresponding importance scores
        :param sg_seqs: skipgram sequences for a given instance
        :param sg_scores: skipgram scores for a given instance
        :param k: the number of top elements to retrieve
        :return: top k sequences and corresponding scores
        """
        # np.argsort gives us the indices of elements from the original array that are required to sort it.
        # taking a subset of -k elements give us the last k indices of elements to sort it.
        # reversing the indices of the last k elements, we get the index order to sort array in descending order.
        # using absolute scores to get top features; ignoring direction of effect
        idx = np.argsort(abs(np.array(scores)))[-k:][::-1]
        top_seqs = list(itemgetter(*idx)(seqs))
        top_scores = list(itemgetter(*idx)(scores))
        return top_seqs, top_scores

    def populate_vocab(self, sg_seqs, sg_scores, max_vocab_size):
        self.vocab = SkipGramVocab.create_vocab(sg_seqs, sg_scores, max_vocab_size)

    def get_sg_bag(self, seqs, scores, min_n, max_n, skip, simplify='discretize'):

        np.seterr(over='raise', under='raise')

        sg_bag = np.zeros(shape=(len(seqs), len(self.vocab)), dtype=np.float128)  # 64 bit causes underflow

        try:
            for i, (cur_seq, cur_score) in enumerate(zip(seqs, scores)):
                cur_inst_sg_seqs, cur_inst_sg_scores = self.get_sg(cur_seq, cur_score, min_n, max_n, skip)
                for sg, score in zip(cur_inst_sg_seqs, cur_inst_sg_scores):
                    if sg in self.vocab.term2idx:
                        # @todo: check if addition of importance is ideal for repeated skipgrams, if any
                        sg_bag[i, self.vocab.term2idx[sg]] += score
        except FloatingPointError as err:
            print(err)

        if simplify == 'sign':
            # convert to ternary values
            sg_bag = np.sign(sg_bag)
        elif simplify == 'discretize':  # discretize scores into 5 bins
            sg_bag = self.discretize_imp(sg_bag)

        sg_bag = sg_bag.astype('int')

        self.sg_bag = sg_bag

    def discretize_imp(self, sg_bag):
        """
        Convert importance scores into discrete values -2, -1, 0, 1 and 2.
        -2 and 2 represent high negative and positive importance respectively.
        -1 and 1 represent low negative and postive importance respectively.
        :param sg_bag: 2D array with importance scores (n_inst * sg_vocab_size)
        :return: Discretized skipgram bag
        """
        for i, row in enumerate(sg_bag):
            for j, elt in enumerate(row):
                if elt < self.neg_th:
                    sg_bag[i, j] = -2
                elif self.neg_th <= elt < 0:
                    sg_bag[i, j] = -1
                elif self.pos_th > elt > 0:
                    sg_bag[i, j] = 1
                elif elt >= self.pos_th:
                    sg_bag[i, j] = 2

        return sg_bag

class SkipGramVocab:

    def __init__(self):
        self.term2idx = dict()
        self.idx2word = dict()

    @classmethod
    def create_vocab(cls, seqs, scores=None, max_vocab_size=None, vocab_filter='kbest'):
        """
        Construct vocabulary from the top scoring sequence skipgrams
        :param seqs: top scoring sequence skipgrams, 2D list n_inst * n_top_sg
        :param scores: scores of all skipgrams, 2D list n_inst * n_top_sg
        :param max_vocab_size: Maximum number of vocab elements to keep
        :param vocab_filter: which filter to use for reducing vocabulary size (freq|kbest)
        :return: an object with a dictionary mapping terms to vocab indices,
                                a dictionary mapping vocab indices to terms,
                                a dictionary mapping vocab terms to their instance frequency
        """

        inst = cls()

        vocab_set = inst._reduce_vocab_size(seqs, scores, max_vocab_size, vocab_filter)

        term2idx = dict()
        for term in vocab_set:
            term2idx.setdefault(term, len(term2idx))

        idx2term = {value: key for key, value in term2idx.items()}

        inst.term2idx = term2idx
        inst.idx2word = idx2term

        return inst

    def _reduce_vocab_size(self, seqs, scores, max_vocab_size, vocab_filter):

        if max_vocab_size is not None:

            print("Keeping top {} vocab items only".format(max_vocab_size))

            if vocab_filter == 'freq':
                term2freq = Counter(x for xs in seqs for x in set(xs))  # number of instances the term occurs in
                print("Original vocab size: ", len(term2freq))
                print("Selecting most frequent top skipgrams")
                vocab_set = dict(term2freq.most_common(max_vocab_size)).keys()  # frequency filter.

            elif vocab_filter == 'kbest':
                print("Selecting most contributing top skipgrams")
                term2score = defaultdict(float)  # records total importance of sg in dataset
                for cur_seq, cur_scores in zip(seqs, scores):
                    for cur_term, cur_score in zip(cur_seq, cur_scores):
                        term2score[cur_term] += abs(cur_score)  # using absolute values and ignoring sign
                term2score.default_factory = None  # turn off default behaviour of defaultdict
                vocab_set = get_top_items_dict(term2score, max_vocab_size, order=False).keys()

        else:
            vocab_set = {x for xs in seqs for x in xs}

        return vocab_set

    def __len__(self):
        return len(self.term2idx)
