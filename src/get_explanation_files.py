import sys
sys.path.append('/home/madhumita/PycharmProjects/rnn_expl_rules/')

from src.corpus_utils import DataUtils, Corpus, CorpusEncoder
from src.classifiers.lstm import LSTMClassifier
from src.classifiers.gru import GRUClassifier
from src.explanations.grads import Explanation
from src.explanations.imp_sg import SeqImpSkipGram
from src.explanations.expl_orig_input import SeqSkipGram
from src.weka_utils.vec_to_arff import get_feat_dict, write_arff_file

from os.path import exists, realpath, join
import resource

soft, hard = 5.4e+10, 5.4e+10  # nearly 50GB
resource.setrlimit(resource.RLIMIT_AS, (soft, hard))

PATH_DIR_CORPUS = '/home/madhumita/sepsis_mimiciii/text/'
FNAME_LABELS = 'sepsis_labels.json'
PATH_DIR_LABELS = '/home/madhumita/sepsis_mimiciii/labels/'
PATH_DIR_SPLITS = '/home/madhumita/sepsis_mimiciii/splits/'

FNAME_ENCODER = 'corpus_encoder_mimiciii.json'
PATH_DIR_ENCODER = '../out/'

model_name = 'lstm'  # lstm|gru
test_mode = 'test'  # val | test

baseline = False


def load_model():
    if model_name == 'lstm':
        classifier = LSTMClassifier.load(
            f_model='sepsis_mimic_lstm_classifier_hid100_emb100.tar')
    elif model_name == 'gru':
        classifier = GRUClassifier.load(
            f_model='sepsis_mimic_gru_classifier_hid50_emb50.tar')
    else:
        raise ValueError("Model should be either 'gru' or 'lstm'")

    return classifier


def load_corpora():
    train_split, val_split, test_split = DataUtils.read_splits(PATH_DIR_SPLITS)

    # initialize corpora
    train_corp = Corpus(PATH_DIR_CORPUS, FNAME_LABELS, PATH_DIR_LABELS, train_split, 'train')
    val_corp = Corpus(PATH_DIR_CORPUS, FNAME_LABELS, PATH_DIR_LABELS, val_split, 'val')
    test_corp = Corpus(PATH_DIR_CORPUS, FNAME_LABELS, PATH_DIR_LABELS, test_split, 'test')

    if not exists(realpath(join(PATH_DIR_ENCODER, FNAME_ENCODER))):
        raise FileNotFoundError("Encoder not found")
    # load encoder
    corpus_encoder = CorpusEncoder.from_json(FNAME_ENCODER, PATH_DIR_ENCODER)

    return train_corp, val_corp, test_corp, corpus_encoder


def write_baseline_expl_files(classifier, train_corp, val_corp, test_corp, corpus_encoder):

    if baseline:
        train_sg = get_sg_baseline(train_corp, classifier, corpus_encoder)
        val_sg = get_sg_baseline(val_corp, classifier, corpus_encoder, vocab=train_sg.vocab)
        test_sg = get_sg_baseline(test_corp, classifier, corpus_encoder, vocab=train_sg.vocab)
        print("Populated files for baseline evaluation")


def get_sg_baseline(eval_corp, classifier, encoder, n_sg=50, vocab=None, max_vocab_size=5000):
    seqs = encoder.get_decoded_sequences(eval_corp)
    y_pred, __ = classifier.predict(eval_corp, encoder)

    min_n, max_n, skip = 1, 4, 2

    sg = SeqSkipGram.from_seqs(seqs, min_n=min_n, max_n=max_n, skip=skip,
                               topk=n_sg, vocab=vocab, max_vocab_size=max_vocab_size)

    # write as arff file
    feat_dict = get_feat_dict(sg.vocab.term2idx, vec_type='binary')
    rel_name = classifier.model_type \
               + '_hid' + str(classifier.hidden_dim) \
               + '_emb' + str(classifier.emb_dim) + '_mimic' \
               + '_min' + str(min_n) + '_max' + str(max_n) + '_skip' \
               + str(skip) + '_baseline_'
    write_arff_file(rel_name,
                    feat_dict, eval_corp.label_encoder.classes_,
                    sg.sg_bag, y_pred,
                    '../out/weka/', rel_name + eval_corp.subset_name + '_pred.arff')

    return sg


def write_explanation_files(classifier, train_corp, val_corp, test_corp, corpus_encoder):

    # populating weka files for interpretability
    train_sg = get_imp_sg(train_corp, classifier, corpus_encoder)
    val_sg = get_imp_sg(val_corp, classifier, corpus_encoder,
                    vocab=train_sg.vocab, pos_th=train_sg.pos_th, neg_th=train_sg.neg_th)
    test_sg = get_imp_sg(test_corp, classifier, corpus_encoder,
                     vocab=train_sg.vocab, pos_th=train_sg.pos_th, neg_th=train_sg.neg_th)

    print("Populated files for getting explanations")


def get_imp_sg(eval_corp, classifier, encoder,
               n_sg=50,
               vocab=None, max_vocab_size=5000, pos_th=0., neg_th=0.,
               get_imp=True):

    explanations = dict()
    method = 'dot'

    if get_imp:
        print("Computing word importance scores")
        explanation = Explanation.get_grad_importance(method, classifier, eval_corp, encoder)
        explanations[method] = explanation
    else:
        print("Loading word importance scores")
        explanations[method] = Explanation.from_imp(method, classifier, eval_corp, encoder)

    print("Getting top skipgrams for subset {}".format(eval_corp.subset_name))
    seqs = encoder.get_decoded_sequences(eval_corp)

    min_n, max_n, skip = 1, 4, 2

    sg = SeqImpSkipGram.from_seqs(seqs, explanations['dot'].imp_scores,
                                  min_n=min_n, max_n=max_n, skip=skip,
                                  topk=n_sg, vocab=vocab, max_vocab_size=max_vocab_size,
                                  pos_th=pos_th, neg_th=neg_th)

    # write as arff file
    feat_dict = get_feat_dict(sg.vocab.term2idx, vec_type='discretized')
    rel_name = classifier.model_type \
               + '_hid' + str(classifier.hidden_dim) \
               + '_emb' + str(classifier.emb_dim) + '_mimic' \
               + '_min' + str(min_n) + '_max' + str(max_n) + '_skip' + str(skip) + '_'
    write_arff_file(rel_name,
                    feat_dict, eval_corp.label_encoder.classes_,
                    sg.sg_bag, explanations['dot'].preds,
                    '../out/weka/', rel_name + eval_corp.subset_name + '_pred.arff')

    return sg


def main():
    classifier = load_model()
    train_corpus, val_corpus, test_corpus, corpus_encoder = load_corpora()

    write_baseline_expl_files(classifier,
                              train_corpus, val_corpus, test_corpus, corpus_encoder)

    write_explanation_files(classifier, train_corpus, val_corpus, test_corpus, corpus_encoder)


if __name__ == '__main__':
    main()

