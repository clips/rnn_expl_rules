from src.corpus_utils import CorpusEncoder, CSVCorpus
from src.classifiers.lstm import LSTMClassifier

from anchor import anchor_text
from anchor import utils
import spacy
from spacy.tokenizer import Tokenizer
import pandas as pd

from os.path import join, splitext
import numpy as np
import pickle


class AnchorExp:

    def __init__(self, model, encoder, class_names, tokenize_on_space,
                 use_unk=False,
                 unk='<unk>',
                 exp_th=0.95,
                 seed=1):
        """
        :param use_unk: If False, replaces words by similar words instead of UNKs
        :param unk: the symbol to use for unknown words
        """
        self.model = model
        self.encoder = encoder
        self.use_unk = use_unk
        self.unk = unk
        self.threshold = exp_th

        # need to install this spacy module separately to enable word similarity
        self.nlp = spacy.load("en_core_web_lg")
        if tokenize_on_space:
            self.nlp.tokenizer = Tokenizer(self.nlp.vocab)
        else:
            self.nlp.tokenizer = self.nlp.Defaults.create_tokenizer(self.nlp)

        np.random.seed(seed)
        self.explainer = anchor_text.AnchorText(self.nlp, class_names,
                                                use_unk_distribution=self.use_unk,
                                                mask_string=self.unk)

    def predict(self, texts):
        """
        First converts strings to token lists (2D, all tokens for all instances),
        and then returns a model's prediction for this list.
        :param texts: list of text strings to make prediction for
        :return:
        """
        texts = [[cur_token.text for cur_token in self.nlp(cur_text)]
                 for cur_text in texts]
        return self.model.predict_from_insts(texts, self.encoder)

    def get_anchors(self, text):
        # Getting output prediction class names for text
        pred = self.explainer.class_names[self.predict([text])[0]]
        print("Prediction:", pred)
        exp = self.explainer.explain_instance(text, self.predict,
                                              threshold=self.threshold)

        if len(exp.names()) == 0:
            print("No anchors found!")
        else:
            # alternative = self.explainer.class_names[1 - self.predict([text])[0]]

            print('Anchor: %s' % (' AND '.join(exp.names())))
            print('Precision: %.2f' % exp.precision())
            # print()
            # print('Examples where anchor applies and model predicts %s:' % pred)
            # print()
            # print('\n'.join([x[0] for x in exp.examples(only_same_prediction=True)]))
            # print()
            # print('Examples where anchor applies and model predicts %s:' % alternative)
            # print()
            # print('\n'.join([x[0] for x in exp.examples(partial_index=0,
            #                                             only_different_prediction=True)]))

            # print('Partial anchor: %s' % (' AND '.join(exp.names(0))))
            # print('Precision: %.2f' % exp.precision(0))
            # print()
            # print('Examples where anchor applies and model predicts %s:' % pred)
            # print()
            # print('\n'.join([x[0] for x in exp.examples(partial_index=0,
            #                                             only_same_prediction=True)]))
            # print()
            # print('Examples where anchor applies and model predicts %s:' % alternative)
            # print()
            # print('\n'.join([x[0] for x in exp.examples(partial_index=0,
            #                                             only_different_prediction=True)]))

        return exp

    def submodular_anchor_precrecall(self, exp_names, exp_precs,
                                     val_one_hot, preds_val,
                                     test_one_hot, preds_test, k=10):
        # returns picked, precisions, recalls
        picked = utils.greedy_pick_anchor(
            exp_names, exp_precs,
            val_one_hot,
            self.encoder.vocab.word2idx,
            k=k,
            threshold=1.1)
        print("Picked anchors: ", picked)
        precs = []
        recs = []
        for i in range(1, k + 1):
            exs = picked[:i]
            anchor_names = [exp_names[i] for i in exs]
            anchor_precs = [exp_precs[i] for i in exs]
            data_anchors = val_one_hot[exs]
            pred_anchors = preds_val[exs]
            prec, rec = utils.evaluate_anchor(
                anchor_names, anchor_precs, data_anchors,
                self.encoder.vocab.word2idx, pred_anchors,
                test_one_hot, preds_test,
                threshold=1.1)
            precs.append(prec)
            recs.append(rec)
        return picked, precs, recs


def get_one_hot(corp, encoder):
    one_hot = np.zeros((len(corp.row_ids), len(encoder.vocab.word2idx)))
    for i, (cur_txt, __) in enumerate(iter(corp)):
        for cur_token in cur_txt:
            try:
                one_hot[i, encoder.vocab.word2idx[cur_token]] = 1
            except KeyError:
                one_hot[i, encoder.vocab.word2idx['<unk>']] = 1

    return one_hot


def get_anchor_exps(ds, space_tokenizer):
    # train_corp = CSVCorpus(ds.FNAME_TRAIN, ds.PATH_DIR_CORPUS, 'train',
    #                        ds.TOKENIZER, ds.LABEL_DICT)

    val_corp = CSVCorpus(ds.FNAME_VAL, ds.PATH_DIR_CORPUS, 'val',
                         ds.TOKENIZER, ds.LABEL_DICT)

    test_corp = CSVCorpus(ds.FNAME_TEST, ds.PATH_DIR_CORPUS, 'test',
                          ds.TOKENIZER, ds.LABEL_DICT)

    encoder = CorpusEncoder.from_json(ds.FNAME_ENCODER, ds.PATH_ENCODER)

    f_model = splitext(ds.FNAME_TRAIN)[0][6:] + '_' + ds.model_name + '_' + \
              str(ds.n_layers) + 'layer' + \
              '_hid' + str(ds.n_hid) + \
              '_emb' + str(ds.n_emb) + \
              '_dropout' + str(ds.dropout) + \
              '_bidir' + str(ds.bidir) + '.tar'

    print("Loading model", f_model)

    classifier = LSTMClassifier.load(
        f_model=f_model, dir_model=ds.PATH_DIR_OUT)

    class_names = sorted(ds.LABEL_DICT, key=ds.LABEL_DICT.get)

    anchor_obj = AnchorExp(classifier, encoder, class_names, space_tokenizer,
                           use_unk=False)

    explanation_names = list()
    explanation_precs = list()

    file_counter = 1
    for i, cur_chunk in enumerate(pd.read_csv(join(val_corp.dir_in,
                                                   val_corp.fname),
                             usecols=['text', 'label'], chunksize=1)):
        text = cur_chunk['text'].iloc[0].lower()
        label = cur_chunk['label'].iloc[0]
        print("Explaining instance:", text)
        print("True label:", class_names[label])
        cur_exp = anchor_obj.get_anchors(text)
        explanation_names.append(cur_exp.names())
        explanation_precs.append(cur_exp.exp_map['precision'])

        if i and i % 200 == 0:
            with open(join(ds.PATH_DIR_OUT,
                           splitext(ds.FNAME_TRAIN)[0][6:] + '_names' +
                           str(file_counter) + '.pkl'), 'wb') as f:
                pickle.dump(explanation_names, f)

            with open(join(ds.PATH_DIR_OUT,
                           splitext(ds.FNAME_TRAIN)[0][6:] + '_precs' +
                           str(file_counter) + '.pkl'), 'wb') as f:
                pickle.dump(explanation_precs, f)

            explanation_names = list()
            explanation_precs = list()
            file_counter += 1

    if len(explanation_names) > 0 or len(explanation_precs) > 0:
        with open(join(ds.PATH_DIR_OUT,
                       splitext(ds.FNAME_TRAIN)[0][6:] + '_names' +
                       str(file_counter) + '.pkl'), 'wb') as f:
            pickle.dump(explanation_names, f)

        with open(join(ds.PATH_DIR_OUT,
                       splitext(ds.FNAME_TRAIN)[0][6:] + '_precs' +
                       str(file_counter) + '.pkl'), 'wb') as f:
            pickle.dump(explanation_precs, f)

        file_counter += 1

    val_one_hot = get_one_hot(val_corp, encoder)
    test_one_hot = get_one_hot(test_corp, encoder)

    preds_val, __ = classifier.predict(val_corp, encoder)
    preds_test, __ = classifier.predict(test_corp, encoder)

    preds_val = np.array(preds_val)
    preds_test = np.array(preds_test)

    explanation_names = list()
    explanation_precs = list()
    for i in range(1, file_counter):
    # for i in [1, 2, 3, 4, 5]:
        with open(join(ds.PATH_DIR_OUT,
                       splitext(ds.FNAME_TRAIN)[0][6:] + '_names' +
                       str(i) + '.pkl'), 'rb') as f:
            cur_names = pickle.load(f)
            explanation_names.extend(cur_names)
        with open(join(ds.PATH_DIR_OUT,
                       splitext(ds.FNAME_TRAIN)[0][6:] + '_precs' +
                       str(i) + '.pkl'), 'rb') as f:
            cur_precs = pickle.load(f)
            explanation_precs.extend(cur_precs)

    print("Total number of loaded exp names: ", len(explanation_names))
    print("Total number of prec: ", len(explanation_precs))

    picked, precs, recs = anchor_obj.submodular_anchor_precrecall(explanation_names,
                                            explanation_precs,
                                            val_one_hot, preds_val,
                                            test_one_hot, preds_test)

    print("Picked anchors: ")
    for i in picked:
        print(i, ":", explanation_names[i])

    print("Precision: ", precs[-1])
    print("Recall: ", recs[-1])
