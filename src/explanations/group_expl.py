from gensim.models import ldamodel
from gensim.corpora import Dictionary
from gensim.parsing.preprocessing import STOPWORDS

from string import punctuation


class ImpTopics:

    def __init__(self, model=None):
        self.model = model

    def get_imp_appended_seqs(self, seqs, scores):
        new_seqs = []
        for cur_seq, cur_scores in zip(seqs, scores):
            modified_seq = []
            for word, score in zip (cur_seq, cur_scores):
                if word in punctuation or word in STOPWORDS:
                    continue
                if score < 0.:
                    word = word + '_neg'
                elif score > 0.:
                    word = word + '_pos'
                else:
                    word = word + '_zero'
                modified_seq.append(word)
            new_seqs.append(modified_seq)
        return new_seqs

    def get_topic_model(self, seqs, n_topics=100):
        id2word = Dictionary(seqs)
        corpus = [id2word.doc2bow(seq) for seq in seqs]
        self.model = ldamodel.LdaModel(corpus, id2word=id2word, num_topics=n_topics)

    def get_topic_words(self, k=5):

        for topic_id in range(self.model.num_topics):
            topk = self.model.show_topic(topic_id, k)
            print(topk)
            topk_words = [w for w, _ in topk]

            # print('{}: {}'.format(topic_id, ' '.join(topk_words)))

    @classmethod
    def from_seqs_scores(cls, seqs, scores):
        inst = cls()

        new_seqs = inst.get_imp_appended_seqs(seqs, scores)
        inst.get_topic_model(new_seqs)

        return inst
