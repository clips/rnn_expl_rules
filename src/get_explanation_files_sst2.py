import sys
sys.path.append('/home/madhumita/PycharmProjects/rnn_expl_rules/')

from src.corpus_utils import CSVCorpus, CorpusEncoder, spacy_eng_tokenizer
from src.classifiers.lstm import LSTMClassifier
from src.explanations.grads import Explanation
from src.explanations.imp_sg import SeqImpSkipGram
from src.explanations.expl_orig_input import SeqSkipGram
from src.weka_utils.vec_to_arff import get_feat_dict, write_arff_file

from os.path import exists, realpath, join
import resource

soft, hard = 5.4e+10, 5.4e+10  # nearly 50GB
resource.setrlimit(resource.RLIMIT_AS, (soft, hard))


PATH_DIR_CORPUS = '../dataset/sst2/'
FNAME_TRAIN = 'train_binary_sent.csv'
FNAME_VAL = 'dev_binary_sent.csv'
FNAME_TEST = 'test_binary_sent.csv'

FNAME_ENCODER = 'corpus_encoder_sentiment.json'
PATH_DIR_ENCODER = '../out/'

model_name = 'lstm'  # lstm|gru
test_mode = 'test'  # val | test

baseline = False


def load_model():
    classifier = LSTMClassifier.load(
        f_model='sentiment_lstm_classifier_hid150_emb300.tar')

    return classifier


def load_corpora():

    # initialize corpora
    label_dict = {'positive': 1, 'negative': 0}
    train_corp = CSVCorpus(FNAME_TRAIN, PATH_DIR_CORPUS, True, 'train',
                           spacy_eng_tokenizer, label_dict)
    val_corp = CSVCorpus(FNAME_VAL, PATH_DIR_CORPUS, True, 'val',
                         spacy_eng_tokenizer, label_dict)
    test_corp = CSVCorpus(FNAME_TEST, PATH_DIR_CORPUS, True, 'test',
                          spacy_eng_tokenizer, label_dict)

    if not exists(realpath(join(PATH_DIR_ENCODER, FNAME_ENCODER))):
        raise FileNotFoundError("Encoder not found")
    # load encoder
    corpus_encoder = CorpusEncoder.from_json(FNAME_ENCODER, PATH_DIR_ENCODER)

    return train_corp, val_corp, test_corp, corpus_encoder


def write_baseline_expl_files(classifier, train_corp, val_corp, test_corp, corpus_encoder):

    train_sg = get_sg_baseline(train_corp, classifier, corpus_encoder)
    val_sg = get_sg_baseline(val_corp, classifier, corpus_encoder,
                             vocab=train_sg.vocab)
    test_sg = get_sg_baseline(test_corp, classifier, corpus_encoder,
                              vocab=train_sg.vocab)
    print("Populated files for baseline evaluation")


def get_sg_baseline(eval_corp, classifier, encoder, n_sg=50,
                    vocab=None, max_vocab_size=5000):
    seqs = encoder.get_decoded_sequences(eval_corp)
    y_pred, __ = classifier.predict(eval_corp, encoder)

    min_n, max_n, skip = 1, 4, 2

    sg = SeqSkipGram.from_seqs(seqs, min_n=min_n, max_n=max_n, skip=skip,
                               topk=n_sg, vocab=vocab, max_vocab_size=max_vocab_size)

    # write as arff file
    feat_dict = get_feat_dict(sg.vocab.term2idx, vec_type='binary')
    rel_name = classifier.model_type \
               + '_hid' + str(classifier.hidden_dim) \
               + '_emb' + str(classifier.emb_dim) + '_sst2' \
               + '_min' + str(min_n) + '_max' + str(max_n) + '_skip' \
               + str(skip) + '_baseline_'
    write_arff_file(rel_name,
                    feat_dict, ['negative, positive'],
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
               + '_emb' + str(classifier.emb_dim) + '_sst2' \
               + '_min' + str(min_n) + '_max' + str(max_n) + '_skip' + str(skip) + '_'
    write_arff_file(rel_name,
                    feat_dict, ['negative', 'positive'],
                    sg.sg_bag, explanations['dot'].preds,
                    '../out/weka/', rel_name + eval_corp.subset_name + '_pred.arff')

    return sg


def main():
    classifier = load_model()
    train_corpus, val_corpus, test_corpus, corpus_encoder = load_corpora()

    if baseline:
        write_baseline_expl_files(classifier,
                                train_corpus, val_corpus, test_corpus, corpus_encoder)

    write_explanation_files(classifier, train_corpus, val_corpus, test_corpus, corpus_encoder)


if __name__ == '__main__':
    main()

