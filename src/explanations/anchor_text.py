from src.corpus_utils import CorpusEncoder, Corpus
from src.classifiers.lstm import LSTMClassifier
from src.utils import FileUtils

from anchor import anchor_text
import spacy
from spacy.tokenizer import Tokenizer

from os.path import realpath, join
import numpy as np


class AnchorExp:

    def __init__(self, model, encoder, use_unk, use_prob):
        '''
        :param use_unk: If True, replaces words by similar words instead of UNKS
        '''
        self.model = model
        self.encoder = encoder
        self.use_prob = use_prob
        self.use_unk = use_unk

    def predict(self, texts):
        '''
        :param texts:
        :return:
        '''
        # print("Obtained list of len", len(texts))
        texts = [text.split() for text in texts]
        return self.model.predict_from_insts(texts, self.encoder, self.use_prob)

    def get_anchors(self, text, class_names):
        nlp = spacy.load('en')
        nlp.tokenizer = Tokenizer(nlp.vocab)

        explainer = anchor_text.AnchorText(nlp, class_names, use_unk_distribution=self.use_unk)

        np.random.seed(1)

        #getting output prediction class names for text
        pred = explainer.class_names[self.predict([text])[0]]
        print("Prediction:", pred)

        alternative = explainer.class_names[1 - self.predict([text])[0]]

        exp = explainer.explain_instance(text, self.predict, threshold=0.95, use_proba=use_prob)

        print('Anchor: %s' % (' AND '.join(exp.names())))
        print('Precision: %.2f' % exp.precision())
        print()
        print('Examples where anchor applies and model predicts %s:' % pred)
        print()
        print('\n'.join([x[0] for x in exp.examples(only_same_prediction=True)]))
        print()
        print('Examples where anchor applies and model predicts %s:' % alternative)
        print()
        print('\n'.join([x[0] for x in exp.examples(partial_index=0, only_different_prediction=True)]))

        print('Partial anchor: %s' % (' AND '.join(exp.names(0))))
        print('Precision: %.2f' % exp.precision(0))
        print()
        print('Examples where anchor applies and model predicts %s:' % pred)
        print()
        print('\n'.join([x[0] for x in exp.examples(partial_index=0, only_same_prediction=True)]))
        print()
        print('Examples where anchor applies and model predicts %s:' % alternative)
        print()
        print('\n'.join([x[0] for x in exp.examples(partial_index=0, only_different_prediction=True)]))

if __name__ == '__main__':
    model = LSTMClassifier.load('lstm_classifier_hid50_emb100.tar', '/home/madhumita/PycharmProjects/rnn_expl_rules/out/')
    encoder = CorpusEncoder.from_json('corpus_encoder.json', '/home/madhumita/PycharmProjects/rnn_expl_rules/out/')

    # no anchors obtained on setting it to False. @todo: Probe why
    use_unk = True

    # predict is a function that returns output predictions from the classifier by taking text as input.
    # if use_proba is True, and we return the probilities with predict, the code errors out because it accepts only
    # integers. Hence, setting it to False as default
    use_prob = False

    anchor_obj = AnchorExp(model, encoder, use_unk, use_prob)

    inst_idx = 163
    idx_list = FileUtils.read_list('val_ids.txt', '/home/madhumita/dataset/sepsis_synthetic/splits/')
    text = open(realpath(join('/home/madhumita/dataset/sepsis_synthetic/text/', idx_list[-inst_idx] + '.txt'))).read()
    print("Explaining instance:")
    print(text)

    classes = ['non_septic', 'septic']


    # corpus = Corpus('/home/madhumita/dataset/sepsis_synthetic/text/',
    #                 'sepsis_labels.json', '/home/madhumita/dataset/sepsis_synthetic/labels/',
    #                 idx_list)
    # classes = corpus.label_encoder.classes_
    #
    # labels = FileUtils.read_json('sepsis_labels.json', '/home/madhumita/dataset/sepsis_synthetic/labels/')
    # print(labels['11903'])

    anchors = anchor_obj.get_anchors(text, classes)