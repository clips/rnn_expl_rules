import sys
sys.path.append('/home/madhumita/PycharmProjects/sepsis/')

from src.data_proc.corpus_utils import DataUtils, Corpus, CorpusEncoder
from src.classifiers.lstm import LSTMClassifier

from sklearn.metrics.classification import f1_score

import torch

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

    train_model = True

    if train_model:

        # initialize vocab
        corpus_encoder = CorpusEncoder.from_corpus(train_corp)

        net_params = {'n_layers': 1,
                      'hidden_dim': 100,
                      'vocab_size': corpus_encoder.vocab.size,
                      'embedding_dim': 100,
                      'dropout': 0.,
                      'label_size': 2,
                      'batch_size': 64
                      }

        classifier = LSTMClassifier(**net_params)

        n_epochs = 10
        lr = 0.001
        optimizer = torch.optim.Adam(classifier.parameters(), lr=lr)

        classifier.train_model(train_corp, corpus_encoder, n_epochs, optimizer)
        classifier.save(corpus_encoder)

    else:
        #load model
        classifier, corpus_encoder = LSTMClassifier.load()

    #get predictions
    y_pred, y_true = classifier.predict(val_corp, corpus_encoder)

    #compute scoring metrics
    print(f1_score(y_true=y_true, y_pred=y_pred, average='macro'))


if __name__ == '__main__':
    main()

