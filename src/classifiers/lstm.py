from src.torch_utils import TorchUtils
from src.torch_utils import CustomEmbedding
from src.classifiers.classifier_base import RNNClassifier

import torch.nn as nn
import torch
import torch.nn.functional as F


class LSTMClassifier(RNNClassifier):

    def __init__(self,
                 n_layers,
                 hidden_dim,
                 vocab_size,
                 padding_idx,
                 embedding_dim,
                 dropout,
                 label_size,
                 batch_size):
        super().__init__(batch_size)

        self.model_type = 'lstm'

        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.emb_dim = embedding_dim
        self.dropout = dropout

        self.n_labels = label_size

        self.hidden_in = self.init_hidden()  # initialize cell states

        # self.word_embeddings = nn.Embedding(self.vocab_size, self.emb_dim, padding_idx=padding_idx).to(self.device) #embedding layer, initialized at random
        self.word_embeddings = CustomEmbedding(self.vocab_size, self.emb_dim, padding_idx=padding_idx) #embedding layer, initialized at random

        self.lstm = nn.LSTM(self.emb_dim, self.hidden_dim, num_layers=self.n_layers, dropout=self.dropout) #lstm layers

        self.hidden2label = nn.Linear(self.hidden_dim, self.n_labels) # hidden to output layer

        self.to(self.device)

    def init_hidden(self):
        """
        initializes hidden and cell states to zero for the first input
        """
        h0 = torch.zeros(self.n_layers, self.batch_size, self.hidden_dim).to(self.device)
        c0 = torch.zeros(self.n_layers, self.batch_size, self.hidden_dim).to(self.device)

        return h0, c0

    def forward(self, sentence, sent_lengths, hidden):

        sort, unsort = TorchUtils.get_sort_unsort(sent_lengths)

        embs = self.word_embeddings(sentence).to(self.device)  # word sequence to embedding sequence

        # truncating the batch length if last batch has fewer elements
        cur_batch_len = len(sent_lengths)
        hidden = (hidden[0][:, :cur_batch_len, :].contiguous(), hidden[1][:, :cur_batch_len, :].contiguous())

        # converts data to packed sequences with data and batch size at every time step after sorting them per lengths
        embs = nn.utils.rnn.pack_padded_sequence(embs[:, sort], sent_lengths[sort], batch_first=False)

        # lstm_out: output of last lstm layer after every time step
        # hidden gets updated and cell states at the end of the sequence
        lstm_out, hidden = self.lstm(embs, hidden)
        # pad the sequences again to convert to original padded data shape
        lstm_out, lengths = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=False)
        # embs, __ = nn.utils.rnn.pad_packed_sequence(embs, batch_first=False)

        # unsort batch
        lstm_out = lstm_out[:, unsort]
        hidden = (hidden[0][:, unsort, :], hidden[1][:, unsort, :])
        # use the output of the last LSTM layer at the end of the last valid timestep to predict output
        # If sequence len is constant, using hidden[0] is the same as lstm_out[-1].
        # For variable len seq, use hidden[0] for the hidden state at last valid timestep.
        # Do it for the last hidden layer
        y = self.hidden2label(hidden[0][-1])
        y = F.log_softmax(y, dim=1)

        return y

    def save(self, f_model, dir_model='../out/'):

        net_params = {'n_layers': self.n_layers,
                      'hidden_dim': self.hidden_dim,
                      'vocab_size': self.vocab_size,
                      'padding_idx': self.word_embeddings.padding_idx,
                      'embedding_dim': self.emb_dim,
                      'label_size': self.n_labels,
                      'dropout': self.dropout,
                      'batch_size': self.batch_size
                      }

        # save model state
        state = {
            'net_params': net_params,
            'state_dict': self.state_dict(),
        }

        TorchUtils.save_model(state, f_model, dir_model)

    @classmethod
    def load(cls, f_model, dir_model='../out/'):

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


if __name__ == '__main__':
    lstm = LSTMClassifier(2, 100, 50, 50, 0.5, 2, 2)

    # variable length sequences
    X0 = torch.LongTensor([1, 5, 8, 19, 43])
    X1 = torch.LongTensor([23,44,5,13,1,34,43])
    X = [X1, X0] # sequence needs to be sorted in descending order of length

    X_padded = nn.utils.rnn.pad_sequence(X, batch_first=True)
    # print(X_padded)

    # @todo: create lengths here

    fwd_out = lstm.forward(X_padded, [7,5], lengths, lstm.hidden_in)

    labels = torch.LongTensor([[1,0],[0,1]])

    loss = lstm.loss(fwd_out, labels)
    print(loss)