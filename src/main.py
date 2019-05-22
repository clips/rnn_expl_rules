import sys
sys.path.append('/home/madhumita/PycharmProjects/sepsis/')

from src.corpus_utils import DataUtils, Corpus, CorpusEncoder
from src.classifiers.lstm import LSTMClassifier
from src.classifiers.gru import GRUClassifier
from src.explanations.grads import Explanation
from src.explanations.eval import InterpretabilityEval
from src.explanations.imp_sg import SeqSkipGram
from src.clamp.clamp_proc import Clamp
from src.utils import FileUtils
from src.weka_utils.vec_to_arff import get_feat_dict, write_arff_file

import torch
from sklearn.metrics import f1_score
import numpy as np

from os.path import exists, realpath, join
from os import makedirs
import resource

soft, hard = 5.4e+10, 5.4e+10  # nearly 50GB
resource.setrlimit(resource.RLIMIT_AS, (soft, hard))


def main():
    # dir_corpus = '/home/madhumita/dataset/sepsis_synthetic/text/'
    dir_corpus = '/home/madhumita/sepsis_synthetic/text/'
    dir_clamp = '/home/madhumita/sepsis_synthetic/clamp/'
    f_labels = 'sepsis_labels.json'
    # dir_labels = '/home/madhumita/dataset/sepsis_synthetic/labels/'
    dir_labels = '/home/madhumita/sepsis_synthetic/labels/'

    # get train, val, test splits
    create_split = False
    # dir_splits = '/home/madhumita/dataset/sepsis_synthetic/splits/'
    dir_splits = '/home/madhumita/sepsis_synthetic/splits/'
    if create_split:
        train_split, val_split, test_split = DataUtils.split_data(f_labels, dir_labels, dir_splits)
    else:
        train_split, val_split, test_split = DataUtils.read_splits(dir_splits)

    # initialize corpora
    train_corp = Corpus(dir_corpus, f_labels, dir_labels, train_split)
    val_corp = Corpus(dir_corpus, f_labels, dir_labels, val_split)
    test_corp = Corpus(dir_corpus, f_labels, dir_labels, test_split)

    # train_model = True
    train_model = False
    model_name = 'lstm'  # lstm|gru

    load_encoder = True
    fname_encoder = 'corpus_encoder.json'
    dir_encoder = '../out/'

    if load_encoder:
        if not exists(realpath(join(dir_encoder, fname_encoder))):
            raise FileNotFoundError("Encoder not found")
        # load encoder
        corpus_encoder = CorpusEncoder.from_json(fname_encoder, dir_encoder)
    else:
        # initialize vocab
        corpus_encoder = CorpusEncoder.from_corpus(train_corp)

        if not exists(dir_encoder):
            makedirs(dir_encoder)
        # serialize encoder
        corpus_encoder.to_json(fname_encoder, dir_encoder)

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
        classifier.save(f_model=model_name+'_classifier_hid'+str(net_params['hidden_dim'])+'_emb'+str(net_params['embedding_dim'])+'.tar')

    else:
        # load model
        if model_name == 'lstm':
            classifier = LSTMClassifier.load(f_model='lstm_classifier_hid50_emb100.tar')
        elif model_name == 'gru':
            classifier = GRUClassifier.load(f_model='gru_classifier_hid50_emb100.tar')
        else:
            raise ValueError("Model should be either 'gru' or 'lstm'")

    test_mode = 'test' # val | test
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

    # populating weka files for interpretability
    clamp_obj = Clamp(dir_clamp)
    train_sg = get_sg_bag(clamp_obj, train_corp, classifier, corpus_encoder, model_name, 'train')
    val_sg = get_sg_bag(clamp_obj, val_corp, classifier, corpus_encoder, model_name, 'val',
                        vocab=train_sg.vocab, pos_th=train_sg.pos_th, neg_th=train_sg.neg_th)
    test_sg = get_sg_bag(clamp_obj, test_corp, classifier, corpus_encoder, model_name, 'test',
                         vocab=train_sg.vocab, pos_th=train_sg.pos_th, neg_th=train_sg.neg_th)


def get_sg_bag(clamp_obj, eval_corp, classifier, encoder, model_name, subset, topk=50,
               vocab=None, max_vocab_size=5000, pos_th=0., neg_th=0., search_sg_params=False):

    print("Getting top skipgrams for subset {}".format(subset))

    eval_obj = InterpretabilityEval(eval_corp, clamp_obj)

    # methods = ['l2', 'sum', 'max', 'dot', 'max_mul']
    methods = ['dot']

    explanations = dict()

    for cur_method in methods:
        print("Pooling method: ", cur_method)

        # computing word importance scores
        explanation = Explanation.get_grad_importance(classifier, eval_corp, encoder, cur_method, model_name, subset)
        explanations[cur_method] = explanation
        # eval_obj.avg_acc_from_corpus(explanation.imp_scores, eval_corp, corpus_encoder)

    # getting skipgrams
    seqs = encoder.get_decoded_sequences(eval_corp)

    if search_sg_params:
        # search over best min_n, max_n and skip parameters
        min_n, max_n, skip = sg_param_search(seqs, explanations['dot'].imp_scores, eval_obj)
    else:
        min_n, max_n, skip = 1, 4, 2

    sg = SeqSkipGram.from_seqs(seqs, explanations['dot'].imp_scores, min_n=min_n, max_n=max_n, skip=skip,
                               topk=topk, vocab=vocab, max_vocab_size=max_vocab_size, pos_th=pos_th, neg_th=neg_th)

    # write as arff file
    feat_dict = get_feat_dict(sg.vocab.term2idx, vec_type='discretized')
    rel_name = model_name + '_hid' + str(classifier.hidden_dim) + '_emb' + str(classifier.emb_dim) + '_synthetic' \
               + '_min' + str(min_n) + '_max' + str(max_n) + '_skip' + str(skip) + '_'
    write_arff_file(rel_name, feat_dict, eval_corp.label_encoder.classes_, sg.sg_bag, eval_corp.labels, '../out/weka/',
                    rel_name + subset + '_pred.arff')

    return sg


def sg_param_search(seqs, scores, eval_obj):

    prec = dict()
    best_prec, best_min_n, best_max_n, best_skip = 0., None, None, None

    for min_n in range(1, 5):
        for max_n in range(min_n, min_n + 4):
            for skip in range(11):
                sg = SeqSkipGram.from_seqs(seqs, scores, min_n=1, max_n=max_n, skip=skip, topk=50)
                cur_prec = eval_obj.avg_prec_sg(sg.top_sg_seqs)
                prec[repr((min_n, max_n, skip))] = cur_prec  # converting key to string for JSON serialization

                if cur_prec > best_prec:
                    best_prec, best_min_n, best_max_n, best_skip = cur_prec, min_n, max_n, skip

                print("Average precision at min_n {}, max_n {}, skip {} is: {}".format(min_n, max_n, skip, cur_prec))
                if max_n == 1:
                    break  # all skip values will give the same unigram. Hence iterating over it only once.

    print(
        "Maximum precision {} for min_n {}, max_n {} and skip {}".format(best_prec, best_min_n, best_max_n, best_skip))

    FileUtils.write_json(prec, 'sg_param_search.json', '../out/')

    return best_min_n, best_max_n, best_skip


if __name__ == '__main__':
    main()

