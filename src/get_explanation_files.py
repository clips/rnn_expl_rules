import sys
sys.path.append('/home/madhumita/PycharmProjects/rnn_expl_rules/')

from src.corpus_utils import CSVCorpus, CorpusEncoder
from src.classifiers.lstm import LSTMClassifier
from src.explanations.grads import Explanation
from src.explanations.imp_sg import SeqImpSkipGram
from src.explanations.expl_orig_input import SeqSkipGram
from src.weka_utils.vec_to_arff import get_feat_dict, write_arff_file

from os.path import exists, realpath, join, splitext

import resource
soft, hard = 5.4e+10, 5.4e+10  # nearly 50GB
resource.setrlimit(resource.RLIMIT_AS, (soft, hard))


n_sg = 50
max_vocab_size = 5000
min_n, max_n, skip = 1, 4, 2
load_imp_score = True


def load_model(ds):
    f_model = splitext(ds.FNAME_TRAIN)[0][6:] + '_' + ds.model_name + '_' + \
              str(ds.n_layers) + 'layer' + \
              '_hid' + str(ds.n_hid) + \
              '_emb' + str(ds.n_emb) + \
              '_dropout' + str(ds.dropout) + \
              '_bidir' + str(ds.bidir) + '.tar'

    classifier = LSTMClassifier.load(
        f_model=f_model, dir_model=realpath(ds.PATH_DIR_OUT))
    return classifier


def load_corpora(ds):

    # initialize corpora
    train_corp = CSVCorpus(ds.FNAME_TRAIN, realpath(ds.PATH_DIR_CORPUS), 'train',
                           ds.TOKENIZER, ds.LABEL_DICT)
    val_corp = CSVCorpus(ds.FNAME_VAL, realpath(ds.PATH_DIR_CORPUS), 'val',
                         ds.TOKENIZER, ds.LABEL_DICT)
    test_corp = CSVCorpus(ds.FNAME_TEST, realpath(ds.PATH_DIR_CORPUS), 'test',
                          ds.TOKENIZER, ds.LABEL_DICT)

    if not exists(realpath(join(ds.PATH_ENCODER, ds.FNAME_ENCODER))):
        raise FileNotFoundError("Encoder not found")
    # load encoder
    corpus_encoder = CorpusEncoder.from_json(ds.FNAME_ENCODER,
                                             realpath(ds.PATH_ENCODER))

    return train_corp, val_corp, test_corp, corpus_encoder


def get_sg_baseline(eval_corp, classifier, encoder, label_keys, ds_name, vocab=None):

    seqs = encoder.get_decoded_sequences(eval_corp)
    y_pred, __ = classifier.predict(eval_corp, encoder)

    sg = SeqSkipGram.from_seqs(seqs, min_n=min_n, max_n=max_n, skip=skip,
                               topk=n_sg, vocab=vocab, max_vocab_size=max_vocab_size)

    # write as arff file
    feat_dict = get_feat_dict(sg.vocab.term2idx, vec_type='binary')
    rel_name = classifier.model_type \
               + '_hid' + str(classifier.hidden_dim) \
               + '_emb' + str(classifier.emb_dim) + '_' + ds_name \
               + '_min' + str(min_n) + '_max' + str(max_n) + '_skip' \
               + str(skip) + '_baseline_'
    write_arff_file(rel_name,
                    feat_dict, label_keys,
                    sg.sg_bag, y_pred,
                    '../out/weka/', rel_name + eval_corp.subset_name + '_pred.arff')

    return sg


def write_baseline_expl_files(classifier, train_corp, val_corp, test_corp,
                              corpus_encoder, label_keys, ds_name):

    train_sg = get_sg_baseline(train_corp, classifier, corpus_encoder,
                               label_keys, ds_name)
    val_sg = get_sg_baseline(val_corp, classifier, corpus_encoder,
                             label_keys, ds_name,
                             vocab=train_sg.vocab)
    test_sg = get_sg_baseline(test_corp, classifier, corpus_encoder,
                              label_keys, ds_name,
                              vocab=train_sg.vocab)
    print("Populated files for baseline evaluation")


def get_imp_sg(eval_corp, classifier, encoder, label_keys, ds_name,
               vocab=None, pos_th=0., neg_th=0.):

    explanations = dict()
    method = 'dot'

    if load_imp_score:
        print("Loading word importance scores")
        explanations[method] = Explanation.from_imp(method, classifier, eval_corp,
                                                    encoder)
    else:
        print("Computing word importance scores")
        explanation = Explanation.get_grad_importance(method, classifier, eval_corp, encoder)
        explanations[method] = explanation

    print("Getting top skipgrams for subset {}".format(eval_corp.subset_name))
    seqs = encoder.get_decoded_sequences(eval_corp)
    sg = SeqImpSkipGram.from_seqs(seqs, explanations['dot'].imp_scores,
                                  min_n=min_n, max_n=max_n, skip=skip,
                                  topk=n_sg, vocab=vocab, max_vocab_size=max_vocab_size,
                                  pos_th=pos_th, neg_th=neg_th)

    # write as arff file
    feat_dict = get_feat_dict(sg.vocab.term2idx, vec_type='discretized')
    rel_name = classifier.model_type \
               + '_hid' + str(classifier.hidden_dim) \
               + '_emb' + str(classifier.emb_dim) + '_' + ds_name \
               + '_min' + str(min_n) + '_max' + str(max_n) + '_skip' + str(skip) + '_'
    write_arff_file(rel_name,
                    feat_dict, label_keys,
                    sg.sg_bag, explanations['dot'].preds,
                    '../out/weka/', rel_name + eval_corp.subset_name + '_pred.arff')

    return sg


def write_explanation_files(classifier,
                            train_corp, val_corp, test_corp, corpus_encoder,
                            label_keys, ds_name):

    # populating weka files for interpretability
    train_sg = get_imp_sg(train_corp, classifier, corpus_encoder, label_keys, ds_name)
    val_sg = get_imp_sg(val_corp, classifier, corpus_encoder, label_keys, ds_name,
                        vocab=train_sg.vocab,
                        pos_th=train_sg.pos_th, neg_th=train_sg.neg_th)
    test_sg = get_imp_sg(test_corp, classifier, corpus_encoder, label_keys, ds_name,
                         vocab=train_sg.vocab,
                         pos_th=train_sg.pos_th, neg_th=train_sg.neg_th)

    print("Populated files for getting explanations")


def get_explanation_files(ds, get_baseline_exps):
    classifier = load_model(ds)
    train_corpus, val_corpus, test_corpus, corpus_encoder = load_corpora(ds)
    ds_name = splitext(ds.FNAME_TRAIN)[0][6:]
    label_keys = sorted(ds.LABEL_DICT, key=ds.LABEL_DICT.get)

    if get_baseline_exps:
        write_baseline_expl_files(classifier,
                                  train_corpus, val_corpus, test_corpus,
                                  corpus_encoder, label_keys, ds_name)

    write_explanation_files(classifier,
                            train_corpus, val_corpus, test_corpus,
                            corpus_encoder, label_keys, ds_name)

