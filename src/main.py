import sys
sys.path.append('/home/madhumita/PycharmProjects/sepsis/')

from src.corpus_utils import DataUtils, Corpus, CorpusEncoder
from src.classifiers.lstm import LSTMClassifier
from src.classifiers.gru import GRUClassifier

import torch
from os.path import exists, realpath, join
from os import makedirs
from sklearn.metrics import f1_score

def main():
    # dir_corpus = '/home/madhumita/dataset/sepsis_synthetic/text/'
    dir_corpus = '/home/madhumita/sepsis_synthetic/text/'
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
    model = 'gru' #lstm|gru

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
                      'embedding_dim': 100,
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
        classifier.save()

    else:
        #load model
        if model == 'lstm':
            classifier= LSTMClassifier.load(f_model='lstm_classifier_100.tar')
        elif model == 'gru':
            classifier= GRUClassifier.load(f_model='gru_classifier_50.tar')
        else:
            raise ValueError("Model should be either 'gru' or 'lstm'")

    #get predictions
    # y_pred, y_true = classifier.predict(val_corp, corpus_encoder)

    #compute scoring metrics
    # print(f1_score(y_true=y_true, y_pred=y_pred, average='macro'))

    gradients = classifier.get_importance(val_corp, corpus_encoder)


if __name__ == '__main__':
    main()

