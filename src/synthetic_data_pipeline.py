import sys
sys.path.append('/home/madhumita/PycharmProjects/rnn_expl_rules/')

from src.corpus_utils import ClampedCSVCorpus, CorpusEncoder, dummy_processor
from src.classifiers.lstm import LSTMClassifier
from src.classifiers.gru import GRUClassifier
from src.explanations.grads import Explanation
from src.explanations.eval import InterpretabilityEval
from src.explanations.imp_sg import SeqImpSkipGram
from src.explanations.expl_orig_input import SeqSkipGram
from src.utils import FileUtils
from src.weka_utils.vec_to_arff import get_feat_dict, write_arff_file

import torch
from sklearn.metrics import f1_score

from os.path import exists, realpath, join
from os import makedirs
import resource

soft, hard = 5.4e+10, 5.4e+10  # nearly 50GB
resource.setrlimit(resource.RLIMIT_AS, (soft, hard))


def main():
    PATH_DIR_CORPUS = '../dataset/sepsis_synthetic/'
    FNAME_TRAIN = 'train_synthetic.csv'
    FNAME_VAL = 'val_synthetic.csv'
    FNAME_TEST = 'test_synthetic.csv'
    PATH_DIR_OUT = '../out/'

    load_encoder = True
    FNAME_ENCODER = 'corpus_encoder.json'
    PATH_ENCODER = '../out/'

    # train_model = True
    train_model = False
    model_name = 'lstm'  # lstm|gru

    label_dict = {"non_septic": 0, "septic": 1}

    # initialize corpora
    train_corp = ClampedCSVCorpus(FNAME_TRAIN, realpath(PATH_DIR_CORPUS), 'train',
                                  dummy_processor, label_dict)
    val_corp = ClampedCSVCorpus(FNAME_VAL, realpath(PATH_DIR_CORPUS), 'val',
                                dummy_processor, label_dict)
    test_corp = ClampedCSVCorpus(FNAME_TEST, realpath(PATH_DIR_CORPUS), 'test',
                                 dummy_processor, label_dict)

    if load_encoder:
        if not exists(realpath(join(PATH_ENCODER, FNAME_ENCODER))):
            raise FileNotFoundError("Encoder not found")
        # load encoder
        corpus_encoder = CorpusEncoder.from_json(FNAME_ENCODER, PATH_ENCODER)
    else:
        # initialize vocab
        corpus_encoder = CorpusEncoder.from_corpus(train_corp)

        if not exists(realpath(PATH_ENCODER)):
            makedirs(realpath(PATH_ENCODER))
        # serialize encoder
        corpus_encoder.to_json(FNAME_ENCODER, realpath(PATH_ENCODER))

    if train_model:
        net_params = {'n_layers': 1,
                      'hidden_dim': 50,
                      'vocab_size': corpus_encoder.vocab.size,
                      'padding_idx': corpus_encoder.vocab.pad,
                      'embedding_dim': 50,
                      'dropout': 0.,
                      'label_size': 2,
                      'batch_size': 64
                      }

        if model_name == 'lstm':
            classifier = LSTMClassifier(**net_params)
        elif model_name == 'gru':
            classifier = GRUClassifier(**net_params)
        else:
            raise ValueError("Model should be either 'gru' or 'lstm'")

        n_epochs = 10
        lr = 0.001
        optimizer = torch.optim.Adam(classifier.parameters(), lr=lr)

        classifier.train_model(train_corp, corpus_encoder, n_epochs, optimizer)
        classifier.save(f_model=model_name +
                                '_classifier_hid' +
                                str(net_params['hidden_dim']) +
                                '_emb'+str(net_params['embedding_dim']) +
                                '.tar',
                        dir_model=realpath(PATH_DIR_OUT))

    else:
        # load model
        if model_name == 'lstm':
            classifier = LSTMClassifier.load(f_model='lstm_classifier_hid100_emb100.tar',
                                             dir_model=realpath(PATH_DIR_OUT))
        elif model_name == 'gru':
            classifier = GRUClassifier.load(f_model='gru_classifier_hid50_emb50.tar',
                                            dir_model=realpath(PATH_DIR_OUT))
        else:
            raise ValueError("Model should be either 'gru' or 'lstm'")

    test_mode = 'test'  # val | test
    if test_mode == 'val':
        eval_corp = val_corp
    elif test_mode == 'test':
        eval_corp = test_corp
    else:
        raise ValueError("Specify val|test corpus for evaluation")

    print("Testing on {} data".format(test_mode))
    # get predictions
    y_pred, y_true = classifier.predict(eval_corp, corpus_encoder)
    # compute scoring metrics
    print(f1_score(y_true=y_true, y_pred=y_pred, average='macro'))

    labels = sorted(label_dict, key=label_dict.get)

    print("Resetting the shuffled train corpus to original state")
    train_corp = ClampedCSVCorpus(FNAME_TRAIN, realpath(PATH_DIR_CORPUS), 'train',
                                  dummy_processor, label_dict)

    baseline = False
    if baseline:
        train_sg = get_sg_baseline(train_corp, classifier, corpus_encoder, labels)
        val_sg = get_sg_baseline(val_corp, classifier, corpus_encoder, labels,
                                 vocab=train_sg.vocab)
        test_sg = get_sg_baseline(test_corp, classifier, corpus_encoder, labels,
                                  vocab=train_sg.vocab)
        print("Wrote files for baseline evaluation")

    # populating weka files for interpretability
    train_sg = get_sg(train_corp, classifier, corpus_encoder, labels)
    val_sg = get_sg(val_corp, classifier, corpus_encoder, labels,
                    vocab=train_sg.vocab, pos_th=train_sg.pos_th, neg_th=train_sg.neg_th)
    test_sg = get_sg(test_corp, classifier, corpus_encoder, labels,
                     vocab=train_sg.vocab, pos_th=train_sg.pos_th, neg_th=train_sg.neg_th)


def get_sg(eval_corp, classifier, encoder, labels,
           n_sg=50,
           vocab=None, max_vocab_size=5000, pos_th=0., neg_th=0.,
           compute_imp=False, search_sg_params=False):

    print("Getting top skipgrams for subset {}".format(eval_corp.subset_name))

    # methods = ['l2', 'sum', 'max', 'dot', 'max_mul']
    methods = ['dot']

    explanations = dict()

    for cur_method in methods:
        print("Pooling method: ", cur_method)

        if compute_imp:
            print("Computing word importance scores")
            # computing word importance scores
            explanation = Explanation.get_grad_importance(cur_method, classifier, eval_corp, encoder)
        else:
            print("Loading word importance scores")
            explanation = Explanation.from_imp(cur_method, classifier, eval_corp, encoder)
        explanations[cur_method] = explanation

        eval_obj = InterpretabilityEval(eval_corp)
        eval_obj.avg_acc_from_corpus(explanation.imp_scores, eval_corp, encoder)

    # getting skipgrams
    seqs = encoder.get_decoded_sequences(eval_corp)

    if search_sg_params:
        # search over best min_n, max_n and skip parameters
        min_n, max_n, skip = sg_param_search(seqs, explanations['dot'].imp_scores, eval_obj)
    else:
        min_n, max_n, skip = 1, 4, 2
        # min_n, max_n, skip = 1, 1, 0

    sg = SeqImpSkipGram.from_seqs(seqs, explanations['dot'].imp_scores,
                                  min_n=min_n, max_n=max_n, skip=skip,
                                  topk=n_sg, vocab=vocab, max_vocab_size=max_vocab_size,
                                  pos_th=pos_th, neg_th=neg_th)

    # write as arff file
    feat_dict = get_feat_dict(sg.vocab.term2idx, vec_type='discretized')
    rel_name = classifier.model_type \
               + '_hid' + str(classifier.hidden_dim) \
               + '_emb' + str(classifier.emb_dim) + '_synthetic' \
               + '_min' + str(min_n) + '_max' + str(max_n) + '_skip' + str(skip) + '_'
    write_arff_file(rel_name,
                    feat_dict, labels,
                    sg.sg_bag, explanations['dot'].preds,
                    '../out/weka/', rel_name + eval_corp.subset_name + '_pred.arff')

    return sg


def get_sg_baseline(eval_corp, classifier, encoder, labels,
                    n_sg=50, vocab=None, max_vocab_size=5000):
    seqs = encoder.get_decoded_sequences(eval_corp)
    y_pred, __ = classifier.predict(eval_corp, encoder)

    min_n, max_n, skip = 1, 4, 2

    sg = SeqSkipGram.from_seqs(seqs, min_n=min_n, max_n=max_n, skip=skip,
                               topk=n_sg, vocab=vocab, max_vocab_size=max_vocab_size)

    # write as arff file
    feat_dict = get_feat_dict(sg.vocab.term2idx, vec_type='binary')
    rel_name = classifier.model_type \
               + '_hid' + str(classifier.hidden_dim) \
               + '_emb' + str(classifier.emb_dim) + '_synthetic' \
               + '_min' + str(min_n) + '_max' + str(max_n) + '_skip' + str(skip) + '_baseline_'
    write_arff_file(rel_name,
                    feat_dict, labels,
                    sg.sg_bag, y_pred,
                    '../out/weka/', rel_name + eval_corp.subset_name + '_pred.arff')

    return sg


def sg_param_search(seqs, scores, eval_obj):

    prec = dict()
    best_prec, best_min_n, best_max_n, best_skip = 0., None, None, None

    for min_n in range(1, 5):
        for max_n in range(min_n, min_n + 4):
            for skip in range(11):
                sg = SeqImpSkipGram.from_seqs(seqs, scores,
                                              min_n=1, max_n=max_n, skip=skip,
                                              topk=50)
                cur_prec = eval_obj.avg_prec_sg(sg.top_sg_seqs)
                prec[repr((min_n, max_n, skip))] = cur_prec  # converting key to string for JSON serialization

                if cur_prec > best_prec:
                    best_prec, best_min_n, best_max_n, best_skip = cur_prec, min_n, max_n, skip

                print("Average precision at min_n {}, max_n {}, skip {} is: {}".
                      format(min_n, max_n, skip, cur_prec))
                if max_n == 1:
                    # all skip values will give the same unigram.
                    # Hence iterating over it only once.
                    break

    print(
        "Maximum precision {} for min_n {}, max_n {} and skip {}".
            format(best_prec, best_min_n, best_max_n, best_skip))

    FileUtils.write_json(prec, 'sg_param_search.json', '../out/')

    return best_min_n, best_max_n, best_skip


if __name__ == '__main__':
    main()

