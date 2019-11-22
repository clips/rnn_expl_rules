from src.corpus_utils import CSVCorpus, CorpusEncoder, spacy_eng_tokenizer
from src.classifiers.lstm import LSTMClassifier
from src.utils import EmbeddingUtils, FileUtils

import torch

from sklearn.metrics import f1_score, accuracy_score

from os.path import exists, realpath, join, splitext
from os import makedirs


def process_model(ds):

    train_corp = CSVCorpus(ds.FNAME_TRAIN, realpath(ds.PATH_DIR_CORPUS), 'train',
                           ds.TOKENIZER, ds.LABEL_DICT)
    val_corp = CSVCorpus(ds.FNAME_VAL, realpath(ds.PATH_DIR_CORPUS), 'val',
                         ds.TOKENIZER, ds.LABEL_DICT)
    test_corp = CSVCorpus(ds.FNAME_TEST, realpath(ds.PATH_DIR_CORPUS), 'test',
                          ds.TOKENIZER, ds.LABEL_DICT)

    if ds.load_encoder:
        if not exists(realpath(join(ds.PATH_ENCODER, ds.FNAME_ENCODER))):
            raise FileNotFoundError("Encoder not found")
        # load encoder
        corpus_encoder = CorpusEncoder.from_json(ds.FNAME_ENCODER, ds.PATH_ENCODER)
    else:
        print("Initializing vocabulary")
        corpus_encoder = CorpusEncoder.from_corpus(train_corp)

        if not exists(realpath(ds.PATH_ENCODER)):
            makedirs(realpath(ds.PATH_ENCODER))
        print("Serializing corpus encoder")
        corpus_encoder.to_json(ds.FNAME_ENCODER, realpath(ds.PATH_ENCODER))

    print("Vocab size:", len(corpus_encoder.vocab))

    if ds.train_model:

        if ds.PRETRAINED_EMBS:
            # get embedding weights matrix
            if ds.embs_from_disk:
                print("Loading word embeddings matrix ...")
                weights = FileUtils.read_numpy(ds.FNAME_EMBS_WT, realpath(ds.PATH_DIR_OUT))
            else:
                weights = EmbeddingUtils.get_embedding_weight(ds.FNAME_EMBS,
                                                              realpath(ds.PATH_DIR_EMBS),
                                                              ds.N_DIM_EMBS,
                                                              corpus_encoder.vocab.word2idx)
                print("Saving word embeddings matrix ...")
                FileUtils.write_numpy(weights, ds.FNAME_EMBS_WT, realpath(ds.PATH_DIR_OUT))

            weights = torch.from_numpy(weights).type(torch.FloatTensor)
        else:
            weights = None

        print("Word embeddings loaded!")

        net_params = {'n_layers': ds.n_layers,
                      'hidden_dim': ds.n_hid,
                      'vocab_size': corpus_encoder.vocab.size,
                      'padding_idx': corpus_encoder.vocab.pad,
                      'embedding_dim': ds.n_emb,
                      'emb_weights': weights,
                      'dropout': ds.dropout,
                      'label_size': len(ds.LABEL_DICT.keys()),
                      'batch_size': 64,
                      'bidir': ds.bidir
                      }

        classifier = LSTMClassifier(**net_params)

        n_epochs = 50
        lr = 0.001
        optimizer = torch.optim.Adam(classifier.parameters(), lr=lr)

        classifier.train_model(train_corp, corpus_encoder, n_epochs, optimizer,
                               val_corp)
        classifier.save(f_model=splitext(ds.FNAME_TRAIN)[0][6:] + '_' +
                                ds.model_name + '_' +
                                str(ds.n_layers) + 'layer' +
                                '_hid' +
                                str(ds.n_hid) +
                                '_emb' +
                                str(ds.n_emb) +
                                '_dropout' +
                                str(ds.dropout) +
                                '_bidir' +
                                str(ds.bidir) +
                                '.tar',
                        dir_model=realpath(ds.PATH_DIR_OUT))

    else:
        f_model = splitext(ds.FNAME_TRAIN)[0][6:] + '_' + ds.model_name + '_' + \
                  str(ds.n_layers) + 'layer' + \
                  '_hid' + str(ds.n_hid) + \
                  '_emb' + str(ds.n_emb) + \
                  '_dropout' + str(ds.dropout) + \
                  '_bidir' + str(ds.bidir) + '.tar'

        print("Loading model", f_model)
        classifier = LSTMClassifier.load(
            f_model=f_model, dir_model=realpath(ds.PATH_DIR_OUT))

    if ds.test_mode == 'val':
        eval_corp = val_corp
    elif ds.test_mode == 'test':
        eval_corp = test_corp
    else:
        raise ValueError("Specify val|test corpus for evaluation")

    print("Testing on {} data".format(ds.test_mode))

    # get predictions
    y_pred, y_true = classifier.predict(eval_corp, corpus_encoder)
    # compute scoring metrics
    print("Macro F1 score: ", f1_score(y_true=y_true, y_pred=y_pred, average='macro'))
    print("Accuracy %", accuracy_score(y_true=y_true, y_pred=y_pred) * 100)

