import numpy as np
import os

from src.clamp.clamp_proc import Clamp

SEPSIS_GOLD = {
        'pneumonia', 'empyema', 'meningitis', 'endocarditis', 'infection',
        'hyperthermia' , 'hypothermia',
        'leukocytosis', 'leukopenia',
        'altered mental status',
        'tachycardia',
        'tachypnea',
        'hyperglycemia',
    }

NEG_CUES = set()
def populate_neg_cues(dir_clamp, dir_text):
    clamp_obj = Clamp()
    for fname in os.listdir(dir_clamp):
        rels = clamp_obj.get_relations_neg(fname, dir_clamp, dir_text)
        for cur_rel in rels:
            if cur_rel.entity1.mention.lower() in SEPSIS_GOLD:
                NEG_CUES.add(cur_rel.entity2.mention)

    print("Negation cues populated; Number of unique cue phrases:", len(NEG_CUES))

def update_gold():
    phrase_to_tokens(SEPSIS_GOLD)
    phrase_to_tokens(NEG_CUES)

def phrase_to_tokens(phrase_set):
    for term in list(phrase_set):
        phrase_set.remove(term) #inplace update
        phrase_set.update(term.split()) #inplace update

class InterpretabilityEval:

    def __init__(self, dir_clamp, dir_text, use_neg = True):

        if use_neg:
            populate_neg_cues(dir_clamp, dir_text)

        if not use_neg:
            assert len(NEG_CUES) == 0, "Negation cues are already populated, and hence are wrongly added to gold"

        update_gold() #convert important phrases to tokens
        # set of important features to be treated as gold
        self.gold = SEPSIS_GOLD.union(NEG_CUES)
        print("Total length of gold standard set of important words", len(self.gold))


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

            assert k <= len(sequences), "More features requested compared to seq length"

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

    # def sign_agreement(self, scores, sequences, preds, gold):
    #
    #     #find subset of instances (of all) where preds == gold
    #     #get sign of importance of infection terms of all instances.
    #     #Infection terms: 'pneumonia' and 'empyema', 'meningitis', 'endocarditis', 'infection'.
    #     #among these, pneumonia and empyema should BOTH be non-n
    #
    #     #For every instance, iterate over the terms and check which keyword terms were negated.
    #     #compare sign of negated term and importance of the term for all keyword terms, all instances?

if __name__ == '__main__':
    InterpretabilityEval(dir_clamp='/home/madhumita/dataset/sepsis_synthetic/clamp',
                      dir_text='/home/madhumita/dataset/sepsis_synthetic/text')