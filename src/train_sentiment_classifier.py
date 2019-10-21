import sys
sys.path.append('/home/madhumita/PycharmProjects/sepsis/')

from src.corpus_utils import SST2Corpus, CorpusEncoder
from src.classifiers.lstm import LSTMClassifier

import torch

from spacy.lang.en import English
from sklearn.metrics import f1_score, accuracy_score

from os.path import exists, realpath, join
from os import makedirs
import resource

soft, hard = 5.4e+10, 5.4e+10  # nearly 50GB
resource.setrlimit(resource.RLIMIT_AS, (soft, hard))

PATH_DIR_CORPUS = '../dataset/sst2/'
FNAME_TRAIN = 'train_binary_sent.csv'
FNAME_VAL = 'dev_binary_sent.csv'
FNAME_TEST = 'test_binary_sent.csv'

load_encoder = True
FNAME_ENCODER = 'corpus_encoder_sentiment.json'
PATH_DIR_ENCODER = '../out/'

train_model = True
model_name = 'lstm'  # lstm|gru

# test_mode = 'val'  # val | test
test_mode = 'test'  # val | test


def init_spacy_eng_tokenizer():
    nlp = English()
    tokenizer = nlp.Defaults.create_tokenizer(nlp)
    return tokenizer


def process_model():

    tokenizer = init_spacy_eng_tokenizer()

    train_corp = SST2Corpus(FNAME_TRAIN, PATH_DIR_CORPUS, 'train', tokenizer)
    val_corp = SST2Corpus(FNAME_VAL, PATH_DIR_CORPUS, 'val', tokenizer)
    test_corp = SST2Corpus(FNAME_TEST, PATH_DIR_CORPUS, 'test', tokenizer)

    if load_encoder:
        if not exists(realpath(join(PATH_DIR_ENCODER, FNAME_ENCODER))):
            raise FileNotFoundError("Encoder not found")
        # load encoder
        corpus_encoder = CorpusEncoder.from_json(FNAME_ENCODER, PATH_DIR_ENCODER)
    else:
        # initialize vocab
        corpus_encoder = CorpusEncoder.from_corpus(train_corp)

        if not exists(PATH_DIR_ENCODER):
            makedirs(PATH_DIR_ENCODER)
        # serialize encoder
        corpus_encoder.to_json(FNAME_ENCODER, PATH_DIR_ENCODER)

    print("Vocab size:", len(corpus_encoder.vocab))

    if train_model:
        net_params = {'n_layers': 1,
                      'hidden_dim': 100,
                      'vocab_size': corpus_encoder.vocab.size,
                      'padding_idx': corpus_encoder.vocab.pad,
                      'embedding_dim': 300,
                      'dropout': 0.,
                      'label_size': 2,
                      'batch_size': 64,
                      'bidir': True
                      }

        classifier = LSTMClassifier(**net_params)

        n_epochs = 50
        lr = 0.001
        optimizer = torch.optim.Adam(classifier.parameters(), lr=lr)

        classifier.train_model(train_corp, corpus_encoder, n_epochs, optimizer, val_corp)
        classifier.save(f_model='sentiment_' +
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
            f_model='sentiment_mimic_lstm_classifier_hid150_emb300.tar')

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


def main():
    process_model()


if __name__ == '__main__':
    main()
