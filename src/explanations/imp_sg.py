from nltk import skipgrams
import numpy as np
from operator import itemgetter

class SeqSkipGram:

    # def __init__(self, top_sg_seqs = None, top_sg_scores = None):
    #     self.top_sg_seqs = top_sg_seqs
    #     self.top_sg_scores = top_sg_scores

    @classmethod
    def from_seqs(cls, seqs, scores, min_n, max_n, skip, topk):

        inst = cls()
        sg_seqs, sg_scores = inst.get_sg(seqs, scores, min_n, max_n, skip)
        inst.filter_sg(sg_seqs, sg_scores, topk)

        inst.populate_vocab()

        return inst


    def get_sg(self, seqs, scores, min_n, max_n, skip):
        """
        Return tuple containing skip gram sequence terms and their aggregated scores
        :param seqs: list of list of sequences
        :param scores: list of list of importance scores of sequences
        :param min_n: minimum ngram length
        :param max_n: maximum ngram length
        :param skip: maximum number of skip positions
        :return: skip grams of sequences and corresponding average scores
        """

        sg_seqs, sg_scores = list(), list()
        for cur_seq, cur_score in zip(seqs, scores):
            for n in range(min_n, max_n+1):
                if not n:
                    continue
                if n == 1:
                    sg_seqs.append(cur_seq)
                    sg_scores.append(cur_score)
                    continue
                sg_seqs.append([' '.join(sg) for sg in skipgrams(cur_seq, n=n, k=skip)])
                sg_scores.append([np.mean(sg) for sg in skipgrams(cur_score, n=n, k=skip)])

        return sg_seqs, sg_scores

    def filter_sg(self, sg_seqs, sg_scores, k):
        """
        Get the top k skipgrams for every instance and their corresponding importance scores
        :param sg_seqs: skipgram sequences for every instance
        :param sg_scores: skipgram scores for every instance
        :param k: the number of top elements to retrieve
        :return: top k sequences and corresponding scores
        """
        top_sg_seqs, top_sg_scores = list(), list()

        for cur_sg_seq, cur_sg_score in zip(sg_seqs, sg_scores):

            #np.argsort gives us the indices of elements from the original array that are required to sort it.
            #taking a subset of -k elements give us the last k indices of elements to sort it.
            #reversing the indices of the last k elements, we get the index order to sort array in descending order.
            cur_k = min(k, len(cur_sg_seq))

            #using absolute scores to get top features; ignoring direction of effect
            idx = np.argsort(abs(np.array(cur_sg_score)))[-cur_k:][::-1]

            top_sg_seqs.append(list(itemgetter(*idx)(cur_sg_seq)))
            top_sg_scores.append(list(itemgetter(*idx)(cur_sg_score)))

            print(list(itemgetter(*idx)(cur_sg_seq)))

        self.top_sg_seqs = top_sg_seqs
        self.top_sg_scores = top_sg_scores

    def populate_vocab(self):
        self.vocab = SkipGramVocab.create_vocab(self.top_sg_seqs)

    def seq_to_sg_bag(self):
        sg_bag = np.zeros(shape = (len(self.top_sg_seqs), len(self.vocab)))

        for i, (cur_seq, cur_scores) in enumerate(zip(self.top_sg_seqs, self.top_sg_scores)):
            for sg, score in zip(cur_seq, cur_scores):
                sg_bag[i, self.vocab.word2idx[sg]] = score

        return sg_bag

class SkipGramVocab:

    def __init__(self):
        self.word2idx = dict()
        self.idx2word = dict()

    @classmethod
    def create_vocab(cls, sg_seqs):
        """
        Construct vocabulary from the top scoring sequence skipgrams
        :param top_sg_seqs: top scoring sequence skipgrams
        :return: a dictionary mapping terms to vocab indices and a dictionary mapping vocab indices to terms
        """
        word2idx = dict()

        for seq in sg_seqs:
            for term in seq:
                word2idx.setdefault(term, len(word2idx))

        idx2word = {value:key for key, value in word2idx.items()}

        inst = cls()
        inst.word2idx = word2idx
        inst.idx2word = idx2word

        print("Vocab size:", len(inst))

        return inst

    def __len__(self):
        return len(self.word2idx)