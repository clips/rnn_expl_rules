import numpy as np

SEPSIS_GOLD = {
        'pneumonia', 'empyema', 'meningitis', 'endocarditis', 'infection',
        'hyperthermia' , 'hypothermia',
        'leukocytosis', 'leukopenia',
        'altered', 'mental', 'status',
        'tachycardia',
        'tachypnea',
        'hyperglycemia',
    } #@todo: add negation markers: compiled from CLAMP?

class InterpretabilityEval:

    def __init__(self, gold = SEPSIS_GOLD):
        #list of important features to be treated as gold
        self.gold = gold

    def avg_prec_recall_f1_at_k(self, scores, sequences, k):
        '''
        Computes precision, recall and F1 score at rank k (information retrieval metric) averaged across all the instances.
        :param scores: 2D list n_inst * seq_len with word importance scores
        :param sequences: 2D list n_inst * seq_len with word sequences
        :param k: Rank k to compute metrics at
        :return: Average precision@k, recall@k and F-score@k
        '''

        avg_prec = 0
        avg_recall = 0

        for score_row, seq_row in zip(scores, sequences):

            # find top k element indices in the instance
            top_k_idx = np.argsort(abs(np.array(score_row)))[-k:]
            top_k = [seq_row[i] for i in top_k_idx]

            # compute overlap between top k and gold
            overlap = self.gold.intersection(set(top_k))

            # compute precision at k
            cur_prec = len(overlap) / len(top_k) #how many of the top features are important as per gold.
            #compute recall at k
            cur_recall = len(overlap) / len(self.gold) #how many of the important features in gold are found as top feats

            avg_prec += cur_prec
            avg_recall += cur_recall

        avg_prec = avg_prec / len(scores)
        avg_recall = avg_recall / len(scores)

        macro_f1 = (2*avg_prec*avg_recall) / (avg_prec+avg_recall)

        print("Average precision: {}, Average recall: {}, Macro-F1: {}".format(avg_prec, avg_recall, macro_f1))

        return avg_prec, avg_recall, macro_f1

    def avg_prec_recall_f1_at_k_from_corpus(self, scores, corpus, corpus_encoder, k):

        seq_lst = corpus_encoder.get_decoded_sequences(corpus)
        self.avg_prec_recall_f1_at_k(scores, seq_lst, k)