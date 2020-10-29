import numpy as np

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


class InterpretabilityEval:

    def __init__(self, corpus, use_neg=True, keywords=SEPSIS_GOLD):
        # populating the list of gold important keywords for the instance
        # the conversion to list can be avoided if we use the generator directly later.
        # Might come in handy for larger datasets.
        self.gold_list = list(self.get_inst_gold(corpus, keywords, use_neg))

    def accuracy(self, scores, sequences, debug=False):
        """
        Computes precision, recall and F1 score at rank k (information retrieval metric) averaged across all the instances.
        :param scores: 2D list n_inst * seq_len with word importance scores
        :param sequences: 2D list n_inst * seq_len with word sequences
        :param debug: True to display messages for debugging
        :return: Average precision@k, recall@k and F-score@k
        """
        avg_acc = 0.
        n_inst = 0

        for score_row, seq_row, gold in zip(scores, sequences, self.gold_list):

            # retrieving as many elements as in gold set for an instance
            k = len(gold)

            if k == 0:
                # empty gold set, skip the instances from calculation
                if debug: print("Skipping instance with empty gold set")
                continue

            n_inst += 1

            if debug: print("Retrieving top {} words".format(k))

            # find top k element indices in the instance
            top_k_idx = np.argsort(abs(np.array(score_row)))[-k:]
            top_k = [seq_row[i] for i in top_k_idx]

            # compute overlap between top k and gold
            overlap = gold.intersection(set(top_k))

            # compute accuracy: same as fscore because k = len(gold)
            cur_acc = len(overlap) / len(top_k)  # how many of the top features are important as per gold.

            avg_acc += cur_acc

        if debug: print("Computed scores for {} instances".format(n_inst))

        avg_acc /= n_inst
        avg_acc *= 100
        print("Average accuracy:", avg_acc)

        # return avg_prec, avg_recall, macro_f1

        return avg_acc

    def avg_acc_from_corpus(self, scores, corpus, corpus_encoder):

        seq_lst = corpus_encoder.get_decoded_sequences(corpus)
        self.accuracy(scores, seq_lst)

    def avg_prec_sg(self, skipgrams):
        """
        This function calculates the precision of the terms in the top skipgrams compared to gold list of important terms,
        averaged over all skipgrams for all instances.
        :param skipgrams: 2D list of important skipgrams for every instance. n_inst * n_sg
        :return avg_inst_prec: average precision
        """
        avg_inst_prec = 0.
        n_inst = 0  # counter only over instances that prec is computed for. Skipping 0-gold instances
        for i, sg_list in enumerate(skipgrams):  # iterating over instances
            if len(self.gold_list[i]) == 0:
                continue  # skip the instances without a gold term
            n_inst += 1
            avg_sg_prec = 0.
            for sg in sg_list:  # iterate over all skipgrams of instance
                sg = sg.split()
                overlap = self.gold_list[i].intersection(sg)
                avg_sg_prec += len(overlap) / len(sg)
            avg_sg_prec /= len(sg_list)
            avg_inst_prec += avg_sg_prec  # macro average prec over all skipgrams
        avg_inst_prec /= n_inst  # macro average skip gram prec over all instances
        return avg_inst_prec

    def get_inst_gold(self, corpus, keywords, use_neg):
        """
        Get gold important terms for every instance based on the gold entity mentions and negation cues in that instance.
        :param corpus: corpus object
        :param dir_clamp: clamp file path
        :param keywords: set of gold keywords used for populating the documents
        :param use_neg: True to use negation triggers in gold set.
        :yields set of gold "important" terms for one instance at a time
        """
        for txt, clamp_txt in corpus.get_text_clamptxt():
            gold = set()
            # set of gold keywords present in the instance
            gold.update(self._get_gold_entity_set(clamp_txt, keywords))
            if use_neg:
                # adding set of negation triggers and corresponding negated keywords to gold set
                gold.update(self._get_gold_neg_set(clamp_txt, txt, keywords))
            yield gold

    def _get_gold_entity_set(self, clamp_txt, keywords):
        gold_ents = set()
        entities = Clamp().get_entities_from_text(clamp_txt)
        # Add all entities present in the instance to the gold set
        for cur_ent in entities:
            for i in keywords:
                if i in cur_ent.mention.lower():
                    gold_ents.update(i.split())
                    break  # terminates the inner loop over keywords if the entity matches a keyword

        return gold_ents

    def _get_gold_neg_set(self, clamp_txt, txt, keywords):
        gold_neg = set()
        rels = Clamp().get_relations_neg_from_text(clamp_txt, txt)
        for cur_rel in rels:
            # check if this relation is a negation of one of the keywords
            for i in keywords:
                if i in cur_rel.entity1.mention.lower():
                    # if entity is a keyword used to populate the docs, add entity and negation marker to gold set
                    gold_neg.update(i.split())  # add the entity. Possibly redundant after entity iteration done earlier?
                    gold_neg.update(cur_rel.entity2.mention.split())  # add the negation marker
                    break

        return gold_neg
