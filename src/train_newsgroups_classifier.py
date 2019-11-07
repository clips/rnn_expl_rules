import sys
sys.path.append('/home/madhumita/PycharmProjects/rnn_expl_rules/')

from src.datasets.newsgroups import Newsgroups

from src.corpus_utils import CSVCorpus, CorpusEncoder, spacy_eng_tokenizer
from src.classifiers.lstm import LSTMClassifier
from src.utils import EmbeddingUtils, FileUtils

import torch

from sklearn.metrics import f1_score, accuracy_score

from os.path import exists, realpath, join
from os import makedirs
import resource

soft, hard = 5.4e+10, 5.4e+10  # nearly 50GB
resource.setrlimit(resource.RLIMIT_AS, (soft, hard))

PATH_DIR_CORPUS = '../dataset/newsgroups/'
FNAME_TRAIN = 'train_newsgroups.csv'
FNAME_VAL = 'val_newsgroups.csv'
FNAME_TEST = 'test_newsgroups.csv'
FNAME_LABELDICT = 'newsgroups_labeldict.json'
PATH_DIR_OUT = '../out/'

PATH_DIR_EMBS = '/home/corpora/word_embeddings/'
FNAME_EMBS = 'glove.840B.300d.txt'
N_DIM_EMBS = 300
embs_from_disk = True

load_encoder = True
FNAME_ENCODER = 'corpus_encoder_newsgroups.json'
PATH_ENCODER = '../out/'

train_model = True
model_name = 'lstm'  # lstm|gru

test_mode = 'val'  # val | test
# test_mode = 'test'  # val | test


def process_model():

    label_dict = FileUtils.read_json(FNAME_LABELDICT, PATH_DIR_CORPUS)

    train_corp = CSVCorpus(FNAME_TRAIN, PATH_DIR_CORPUS, True, 'train',
                           spacy_eng_tokenizer, label_dict)
    val_corp = CSVCorpus(FNAME_VAL, PATH_DIR_CORPUS, True, 'val',
                         spacy_eng_tokenizer, label_dict)
    test_corp = CSVCorpus(FNAME_TEST, PATH_DIR_CORPUS, True, 'test',
                          spacy_eng_tokenizer, label_dict)

    if load_encoder:
        if not exists(realpath(join(PATH_ENCODER, FNAME_ENCODER))):
            raise FileNotFoundError("Encoder not found")
        # load encoder
        corpus_encoder = CorpusEncoder.from_json(FNAME_ENCODER, PATH_ENCODER)
    else:
        # initialize vocab
        corpus_encoder = CorpusEncoder.from_corpus(train_corp)

        if not exists(PATH_ENCODER):
            makedirs(PATH_ENCODER)
        # serialize encoder
        corpus_encoder.to_json(FNAME_ENCODER, PATH_ENCODER)

    print("Vocab size:", len(corpus_encoder.vocab))

    # get embedding weights matrix
    if embs_from_disk:
        print("Loading word embeddings matrix ...")
        weights = FileUtils.read_numpy('pretrained_embs_newsgroups.npy', PATH_DIR_OUT)
    else:
        weights = EmbeddingUtils.get_embedding_weight(FNAME_EMBS,
                                                      PATH_DIR_EMBS,
                                                      N_DIM_EMBS,
                                                      corpus_encoder.vocab.word2idx)
        print("Saving word embeddings matrix ...")
        FileUtils.write_numpy(weights, 'pretrained_embs_newsgroups.npy', PATH_DIR_OUT)

    weights = torch.from_numpy(weights).type(torch.FloatTensor)

    print("Word embeddings loaded!")
    if train_model:
        net_params = {'n_layers': 1,
                      'hidden_dim': 75,
                      'vocab_size': corpus_encoder.vocab.size,
                      'padding_idx': corpus_encoder.vocab.pad,
                      'embedding_dim': 300,
                      'emb_weights': weights,
                      'dropout': 0.,
                      'label_size': 20,
                      'batch_size': 64,
                      'bidir': True
                      }

        classifier = LSTMClassifier(**net_params)

        n_epochs = 50
        lr = 0.001
        optimizer = torch.optim.Adam(classifier.parameters(), lr=lr)

        classifier.train_model(train_corp, corpus_encoder, n_epochs, optimizer,
                               val_corp)
        classifier.save(f_model='newsgroups_' +
                                model_name +
                                '_classifier_hid' +
                                str(net_params['hidden_dim']) +
                                '_emb' +
                                str(net_params['embedding_dim']) +
                                '.tar'
                        )

    else:
        # load model
        classifier = LSTMClassifier.load(
            f_model='newsgroups_lstm_classifier_hid150_emb300.tar')

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
    print("Macro F1 score: ", f1_score(y_true=y_true, y_pred=y_pred, average='macro'))
    print("Accuracy %", accuracy_score(y_true=y_true, y_pred=y_pred) * 100)


def main(write_csv):
    if write_csv:
        Newsgroups(cats=None, dir_csv='../dataset/newsgroups/')
    process_model()


if __name__ == '__main__':
    write_csv = False
    main(write_csv)
