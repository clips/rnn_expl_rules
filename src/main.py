from src.data_proc.corpus_utils import DataUtils, Corpus, CorpusEncoder
from src.classifiers.lstm import LSTMClassifier

import torch

def main():
    dir_corpus = '/home/madhumita/dataset/sepsis_synthetic/text/'
    f_labels = 'sepsis_labels.json'
    dir_labels = '/home/madhumita/dataset/sepsis_synthetic/labels/'

    #get train, val, test splits
    create_split = False
    dir_splits = '/home/madhumita/dataset/sepsis_synthetic/splits/'
    if create_split:
        train_split, val_split, test_split = DataUtils.split_data(f_labels, dir_labels, dir_splits)
    else:
        train_split, val_split, test_split = DataUtils.read_splits(dir_splits)

    #initialize corpora
    train_corp = Corpus(dir_corpus, f_labels, dir_labels, train_split)
    val_corp = Corpus(dir_corpus, f_labels, dir_labels, val_split)

    #initialize vocab
    corpus_encoder = CorpusEncoder.from_corpus(train_corp)

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    n_classes = 2

    n_layers = 1
    hidden_dim = 100
    vocab_size = corpus_encoder.vocab.size
    embedding_dim = 200
    dropout = 0.5
    batch_size = 64

    n_epochs = 50


    classifier = LSTMClassifier(n_layers, hidden_dim, vocab_size, embedding_dim, dropout, n_classes, batch_size, device)

    lr = 0.001
    optimizer = torch.optim.Adam(classifier.parameters(), lr=lr)

    classifier.train_model(train_corp, corpus_encoder, n_epochs, optimizer)

if __name__ == '__main__':
    main()

