from nltk import skipgrams
import numpy as np
from operator import itemgetter
from collections import Counter

class SeqSkipGram:

    def __init__(self, vocab):
        self.vocab = vocab

    @classmethod
    def from_seqs(cls, seqs, scores, min_n, max_n, skip, topk, vocab = None, max_vocab_size = None):

        inst = cls(vocab)
        inst.get_sg(seqs, scores, min_n, max_n, skip, topk)
        # inst.filter_sg(sg_seqs, sg_scores, topk)

        if inst.vocab is None:
            inst.populate_vocab(max_vocab_size)

        return inst


    def get_sg(self, seqs, scores, min_n, max_n, skip, topk):
        """
        Return tuple containing skip gram sequence terms and their aggregated scores
        :param seqs: list of list of sequences
        :param scores: list of list of importance scores of sequences
        :param min_n: minimum ngram length
        :param max_n: maximum ngram length
        :param skip: maximum number of skip positions
        :param topk: number of top skipgrams in every instance to retain
        """

        top_sg_seqs, top_sg_scores = list(), list()
        for cur_seq, cur_score in zip(seqs, scores):
            cur_inst_sg_seqs, cur_inst_scores = list(), list()
            for n in range(min_n, max_n+1):
                if not n:
                    continue
                if n == 1:
                    cur_inst_sg_seqs.extend(cur_seq)
                    cur_inst_scores.extend(cur_score)
                    continue

                cur_inst_sg_seqs.extend([' '.join(sg) for sg in skipgrams(cur_seq, n=n, k=skip)])
                cur_inst_scores.extend([np.mean(sg) for sg in skipgrams(cur_score, n=n, k=skip)])

            cur_top_sg_seqs, cur_top_scores = self.get_top_sg(cur_inst_sg_seqs, cur_inst_scores, topk)

            # removing skipgrams not in vocab if we already have a vocab (for val and test cases)
            # may not be needed if we filter later before creating a bag rep
            # if self.vocab is not None:
            #     for cur_sg, cur_score in zip(cur_top_sg_seqs, cur_top_scores):
            #         if cur_sg not in self.vocab.term2idx:
            #             cur_top_sg_seqs.remove(cur_sg)
            #             cur_top_scores.remove(cur_score)

            top_sg_seqs.append(cur_top_sg_seqs)
            top_sg_scores.append(cur_top_scores)

        self.top_sg_seqs = top_sg_seqs
        self.top_sg_scores = top_sg_scores

    def get_top_sg(self, sg_seq, sg_score, k):
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
        idx = np.argsort(abs(np.array(sg_score)))[-k:][::-1]

        top_seqs = list(itemgetter(*idx)(sg_seq))
        top_sg_scores = list(itemgetter(*idx)(sg_score))

        return top_seqs, top_sg_scores

    def populate_vocab(self, max_vocab_size):
        self.vocab = SkipGramVocab.create_vocab(self.top_sg_seqs, max_vocab_size)

    def seq_to_sg_bag(self):

        sg_bag = np.zeros(shape = (len(self.top_sg_seqs), len(self.vocab)))

        for i, (cur_seq, cur_scores) in enumerate(zip(self.top_sg_seqs, self.top_sg_scores)):
            #@todo: check if addition of importance is ideal for repeated skipgrams, if any
            for sg, score in zip(cur_seq, cur_scores):
                if sg in self.vocab.term2idx:
                    sg_bag[i, self.vocab.term2idx[sg]] += score

        return sg_bag

class SkipGramVocab:

    def __init__(self):
        self.term2idx = dict()
        self.idx2word = dict()
        self.term2freq = dict()

    @classmethod
    def create_vocab(cls, sg_seqs, max_vocab_size = None):
        """
        Construct vocabulary from the top scoring sequence skipgrams
        :param sg_seqs: top scoring sequence skipgrams, 2D list n_inst * n_top_sg
        :param max_vocab_size: Maximum number of vocab elements to keep
        :return: an object with a dictionary mapping terms to vocab indices,
                                a dictionary mapping vocab indices to terms,
                                a dictionary mapping vocab terms to their instance frequency
        """
        term2freq = Counter(x for xs in sg_seqs for x in set(xs))  # number of instances the term occurs in
        print("Vocab size: ", len(term2freq))

        if max_vocab_size is not None:
            print("Keeping top {} vocab items only".format(max_vocab_size))
            term2freq = dict(term2freq.most_common(max_vocab_size)) #frequency filter. @todo: filter per class instead?
        else:
            term2freq = dict(term2freq)

        term2idx = dict()

        for seq in sg_seqs:
            for term in seq:
                if term in term2freq:
                    term2idx.setdefault(term, len(term2idx))

        idx2word = {value:key for key, value in term2idx.items()}

        inst = cls()
        inst.term2idx = term2idx
        inst.idx2word = idx2word
        inst.term2freq = term2freq

        # print("Vocab size:", len(inst))

        return inst

    def __len__(self):
        return len(self.term2idx)