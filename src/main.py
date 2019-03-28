import sys
sys.path.append('/home/madhumita/PycharmProjects/sepsis/')

from src.corpus_utils import DataUtils, Corpus, CorpusEncoder
from src.classifiers.lstm import LSTMClassifier
from src.classifiers.gru import GRUClassifier
from src.explanations.grads import Explanation
from src.explanations.eval import InterpretabilityEval
from src.explanations.rules import SeqSkipGram

import torch
from os.path import exists, realpath, join
from os import makedirs
from sklearn.metrics import f1_score

def main():
    # dir_corpus = '/home/madhumita/dataset/sepsis_synthetic/text/'
    dir_corpus = '/home/madhumita/sepsis_synthetic/text/'
    dir_clamp = '/home/madhumita/sepsis_synthetic/clamp/'
    f_labels = 'sepsis_labels.json'
    # dir_labels = '/home/madhumita/dataset/sepsis_synthetic/labels/'
    dir_labels = '/home/madhumita/sepsis_synthetic/labels/'

    #get train, val, test splits
    create_split = False
    # dir_splits = '/home/madhumita/dataset/sepsis_synthetic/splits/'
    dir_splits = '/home/madhumita/sepsis_synthetic/splits/'
    if create_split:
        train_split, val_split, test_split = DataUtils.split_data(f_labels, dir_labels, dir_splits)
    else:
        train_split, val_split, test_split = DataUtils.read_splits(dir_splits)

    #initialize corpora
    train_corp = Corpus(dir_corpus, f_labels, dir_labels, train_split)
    val_corp = Corpus(dir_corpus, f_labels, dir_labels, val_split)

    # train_model = True
    train_model = False
    model = 'lstm' #lstm|gru

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

        if model == 'lstm':
            classifier = LSTMClassifier(**net_params)
        elif model == 'gru':
            classifier = GRUClassifier(**net_params)
        else:
            raise ValueError("Model should be either 'gru' or 'lstm'")

        n_epochs = 10
        lr = 0.001
        optimizer = torch.optim.Adam(classifier.parameters(), lr=lr)

        classifier.train_model(train_corp, corpus_encoder, n_epochs, optimizer)
        classifier.save(f_model=model+'_classifier_hid'+str(net_params['hidden_dim'])+'_emb'+str(net_params['embedding_dim'])+'.tar')

    else:
        #load model
        if model == 'lstm':
            classifier= LSTMClassifier.load(f_model='lstm_classifier_hid50_emb100.tar')
        elif model == 'gru':
            classifier= GRUClassifier.load(f_model='gru_classifier_hid100_emb100.tar')
        else:
            raise ValueError("Model should be either 'gru' or 'lstm'")

    #get predictions
    y_pred, y_true = classifier.predict(val_corp, corpus_encoder)

    #compute scoring metrics
    print(f1_score(y_true=y_true, y_pred=y_pred, average='macro'))

    eval_obj = InterpretabilityEval(dir_clamp, dir_corpus)

    methods = ['dot', 'mod_dot', 'sum', 'l2', 'max', 'max_mul', 'l2_mul']
    # methods = ['dot']

    explanations = dict()

    for cur_method in methods:
        print("Pooling method: ", cur_method)

        # computing word importance scores
        explanation = Explanation.get_grad_importance(classifier, val_corp, corpus_encoder, cur_method, model)
        explanations[cur_method] = explanation

        eval_obj.avg_prec_recall_f1_at_k_from_corpus(explanation.imp_scores, val_corp, corpus_encoder, k=15)


    # explanations = classifier.get_importance(val_corp, corpus_encoder, eval_obj)

    #getting skipgrams
    # seqs = corpus_encoder.get_decoded_sequences(val_corp)
    # sg = SeqSkipGram.from_seqs(seqs, explanations['dot'].imp_scores, min_n = 1, max_n = 4, skip = 3, topk=5)


if __name__ == '__main__':
    main()

