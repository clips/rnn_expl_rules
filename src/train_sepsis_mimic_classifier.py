import sys
sys.path.append('/home/madhumita/PycharmProjects/sepsis/')

from src.corpus_utils import DataUtils, Corpus, CorpusEncoder
from src.classifiers.lstm import LSTMClassifier
from src.classifiers.gru import GRUClassifier

import torch
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix

from os.path import exists, realpath, join
from os import makedirs
import resource

soft, hard = 5.4e+10, 5.4e+10  # nearly 50GB
resource.setrlimit(resource.RLIMIT_AS, (soft, hard))

PATH_DIR_IN = '/home/madhumita/sepsis_mimiciii_discharge/'
PATH_DIR_CORPUS = join(PATH_DIR_IN, 'text/')
PATH_DIR_LABELS = join(PATH_DIR_IN, 'labels/')
FNAME_LABELS = 'sepsis_labels.json'

PATH_DIR_SPLITS = join(PATH_DIR_IN, 'splits/')
create_split = False

load_encoder = True
FNAME_ENCODER = 'corpus_encoder_mimiciii_discharge.json'
PATH_DIR_ENCODER = '../out/'

train_model = False
model_name = 'lstm'  # lstm|gru

test_mode = 'test'  # val | test


def process_model():
    # get train, val, test splits
    if create_split:
        train_split, val_split, test_split = DataUtils.split_data(FNAME_LABELS,
                                                                  PATH_DIR_LABELS,
                                                                  PATH_DIR_SPLITS)
    else:
        train_split, val_split, test_split = DataUtils.read_splits(PATH_DIR_SPLITS)

    # initialize corpora
    train_corp = Corpus(PATH_DIR_CORPUS, FNAME_LABELS, PATH_DIR_LABELS, train_split, 'train')
    val_corp = Corpus(PATH_DIR_CORPUS, FNAME_LABELS, PATH_DIR_LABELS, val_split, 'val')
    test_corp = Corpus(PATH_DIR_CORPUS, FNAME_LABELS, PATH_DIR_LABELS, test_split, 'test')

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

    train_corp.get_class_distribution()

    if train_model:
        net_params = {'n_layers': 2,
                      'hidden_dim': 100,
                      'vocab_size': corpus_encoder.vocab.size,
                      'padding_idx': corpus_encoder.vocab.pad,
                      'embedding_dim': 100,
                      'dropout': 0.,
                      'label_size': 2,
                      'batch_size': 64,
                      'bidir': True
                      }

        if model_name == 'lstm':
            classifier = LSTMClassifier(**net_params)
        elif model_name == 'gru':
            classifier = GRUClassifier(**net_params)
        else:
            raise ValueError("Model should be either 'gru' or 'lstm'")

        n_epochs = 50
        lr = 0.001
        optimizer = torch.optim.Adam(classifier.parameters(), lr=lr)

        classifier.train_model(train_corp, corpus_encoder, n_epochs, optimizer, val_corp)
        classifier.save(f_model='sepsis_mimic_discharge_' +
                                model_name +
                                '_classifier_hid' +
                                str(net_params['hidden_dim']) +
                                '_emb' +
                                str(net_params['embedding_dim']) +
                                '.tar'
                        )

    else:
        # load model
        if model_name == 'lstm':
            classifier = LSTMClassifier.load(
                f_model='sepsis_mimic_discharge_lstm_classifier_hid100_emb100.tar')
        elif model_name == 'gru':
            classifier = GRUClassifier.load(
                f_model='sepsis_mimic_discharge_gru_classifier_hid50_emb50.tar')
        else:
            raise ValueError("Model should be either 'gru' or 'lstm'")

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
    print("Class 1 F1 score: ", f1_score(y_true=y_true, y_pred=y_pred))
    print("Macro F1 score: ", f1_score(y_true=y_true, y_pred=y_pred, average='macro'))
    print("Accuracy %", accuracy_score(y_true=y_true, y_pred=y_pred) * 100)
    print("Confusion matrix: ", confusion_matrix(y_true=y_true, y_pred=y_pred))


def main():
    process_model()


if __name__ == '__main__':
    main()

