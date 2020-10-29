from src.torch_utils import CustomEmbedding
from src.torch_utils import TorchUtils
from src.classifiers.classifier_base import RNNClassifier

import torch.nn as nn
import torch
import torch.nn.functional as F


class GRUClassifier(RNNClassifier):

    def __init__(self,
                 n_layers,
                 hidden_dim,
                 vocab_size,
                 padding_idx,
                 embedding_dim,
                 dropout,
                 label_size,
                 batch_size,
                 bidir=False):

        super().__init__(batch_size)

        self.model_type = 'gru'

        self.n_gru_layers = n_layers
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.emb_dim = embedding_dim
        self.dropout = dropout
        self.n_labels = label_size

        self.hidden_in = self.init_hidden()  # initialize cell states

        # self.word_embeddings = nn.Embedding(self.vocab_size, self.emb_dim, padding_idx = padding_idx).to(self.device) #embedding layer, initialized at random
        self.word_embeddings = CustomEmbedding(self.vocab_size, self.emb_dim,
                                               padding_idx=padding_idx) #embedding layer, initialized at random

        # gru layers
        self.gru = nn.GRU(self.emb_dim, self.hidden_dim,
                          num_layers=self.n_gru_layers,
                          dropout=self.dropout,
                          bidirectional=bidir)

        self.hidden2label = nn.Linear(self.hidden_dim, self.n_labels) #hidden to output layer

        self.to(self.device)

    def init_hidden(self):
        """
        initializes hidden and cell states to zero for the first input
        """
        if self.bidirectional:
            n_dirs = 2
        else:
            n_dirs = 1
        h0 = torch.zeros(self.n_gru_layers*n_dirs, self.batch_size, self.hidden_dim).to(self.device)
        return h0

    def forward(self, sentence, sent_lengths, hidden):

        sort, unsort = TorchUtils.get_sort_unsort(sent_lengths)

        embs = self.word_embeddings(sentence).to(self.device)  # word sequence to embedding sequence

        # truncating the batch length if last batch has fewer elements
        cur_batch_len = len(sent_lengths)
        hidden = hidden[:, :cur_batch_len, :].contiguous()

        # converts data to packed sequences with data and batch size at every time step after sorting them per lengths
        embs = nn.utils.rnn.pack_padded_sequence(embs[:, sort], sent_lengths[sort], batch_first=False)

        # gru_out: output of last gru layer after every time step
        # self.hidden gets the updated hidden and cell states at the end of the sequence
        gru_out, hidden = self.gru(embs, hidden)
        # pad the sequences again to convert to original padded data shape
        gru_out, lengths = nn.utils.rnn.pad_packed_sequence(gru_out, batch_first=False)
        # embs, __ = nn.utils.rnn.pad_packed_sequence(embs, batch_first=False)

        #unsort batch
        gru_out = gru_out[:, unsort]
        hidden = hidden[:, unsort, :]
        # use the output of the last GRU layer at the end of the last valid timestep to predict output
        # If sequence len is constant, using hidden is the same as gru_out[-1].
        # For variable len seq, use hidden for the hidden state at last valid timestep. Do it for the last hidden layer
        y = self.hidden2label(hidden[-1])
        y = F.log_softmax(y, dim=1)

        return y

    def save(self, f_model, dir_model='../out/'):

        net_params = {'n_layers': self.n_gru_layers,
                      'hidden_dim': self.hidden_dim,
                      'vocab_size': self.vocab_size,
                      'padding_idx': self.word_embeddings.padding_idx,
                      'embedding_dim': self.emb_dim,
                      'dropout': self.dropout,
                      'label_size': self.n_labels,
                      'batch_size': self.batch_size,
                      'bidir': self.bidirectional
                      }

        # save model state
        state = {
            'net_params': net_params,
            'state_dict': self.state_dict(),
        }

        TorchUtils.save_model(state, f_model, dir_model)

    @classmethod
    def load(cls, f_model = 'gru_classifier.tar', dir_model = '../out/'):

        state = TorchUtils.load_model(f_model, dir_model)
        classifier = cls(**state['net_params'])
        classifier.load_state_dict(state['state_dict'])
        return classifier

    @property
    def hidden_in(self):
        return self._hidden_in

    @hidden_in.setter
    def hidden_in(self, val):
        self._hidden_in = val
