from src.classifiers.torch_utils import TorchUtils

import torch.nn as nn
import torch
import torch.nn.functional as F

from random import shuffle

class LSTMClassifier(nn.Module):

    def __init__(self, n_layers, hidden_dim, vocab_size, embedding_dim, dropout, label_size, batch_size, device):

        super(LSTMClassifier, self).__init__()

        self.n_lstm_layers = n_layers
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.device = device

        #@todo: initialize from pretrained embeddings
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim) #embedding layer, initialized at random

        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=self.n_lstm_layers, dropout=dropout) #lstm layers
        self.hidden2label = nn.Linear(hidden_dim, label_size) #hidden to output layer
        self.hidden = self.init_hidden() #initialize cell states

    def init_hidden(self):
        '''
        initializes hidden and cell states to zero for the first input
        '''
        h0 = torch.zeros(self.n_lstm_layers, self.batch_size, self.hidden_dim).to(self.device)
        c0 = torch.zeros(self.n_lstm_layers, self.batch_size, self.hidden_dim).to(self.device)

        return (h0, c0)

    def forward(self, sentence, sent_lengths):

        sort, unsort = TorchUtils.get_sort_unsort(sent_lengths)

        embs = self.word_embeddings(sentence)  # word sequence to embedding sequence

        # view reshapes the data to the given dimensions. -1: infer from the rest. We want (seq_len * batch_size * input_size)
        # embs = embeds.view(sentence.shape[0], sentence.shape[1], -1)

        # converts data to packed sequences with data and batch size at every time step after sorting them per lengths
        embs = nn.utils.rnn.pack_padded_sequence(embs[:, sort], sent_lengths[sort], batch_first=False)

        # lstm_out: output of last lstm layer after every time step
        # self.hidden gets the updated hidden and cell states at the end of the sequence
        lstm_out, self.hidden = self.lstm(embs, self.hidden)
        # pad the sequences again to convert to original padded data shape
        lstm_out, lengths = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=False)

        #unsort batch
        lstm_out = lstm_out[:, unsort]
        self.hidden = (self.hidden[0][:, unsort], self.hidden[1][:, unsort])
        # use the output of the last LSTM layer at the end of the last valid timestep to predict output
        # If sequence len is constant, using self.hidden[0] is the same as lstm_out[-1].
        # For variable len seq, use hidden[0] for the hidden state at last valid timestep. Do it for the last hidden layer
        y = self.hidden2label(self.hidden[0][-1])
        y = F.log_softmax(y, dim=1)

        return y

    def loss(self, fwd_out, target):
        #NLL loss to be used when logits have log-softmax output.
        #If softmax layer is not added, directly CrossEntropyLoss can be used.
        loss_fn = nn.NLLLoss()
        return loss_fn(fwd_out, target)

    def train_model(self, corpus, corpus_encoder, n_epochs, optimizer):

        optimizer = optimizer
        for i in range(n_epochs):
            running_loss = 0.0

            #shuffle the corpus
            combined = list(zip(corpus.fname_subset, corpus.labels))
            shuffle(combined)
            corpus.fname_subset, corpus.labels = zip(*combined)

            #get train batch
            for idx, (cur_insts, cur_labels) in enumerate(corpus_encoder.get_batches(corpus, self.batch_size)):
                cur_insts, cur_labels, cur_lengths = corpus_encoder.batch_to_tensors(cur_insts, cur_labels, self.device)

                # forward pass
                fwd_out = self.forward(cur_insts, cur_lengths)

                # loss calculation
                loss = self.loss(fwd_out, cur_labels)

                # backprop
                optimizer.zero_grad()  # reset tensor gradients
                loss.backward()  # compute gradients for network params w.r.t loss
                optimizer.step()  # perform the gradient update step

                self.hidden[0].detach_()
                self.hidden[1].detach_()

                # print statistics
                running_loss += loss.item()

                if idx % 100 == 99:  # print every 100 mini-batches
                    print('[%d, %5d] loss: %.3f' %
                          (i + 1, idx + 1, running_loss / 2000))
                    running_loss = 0.0



if __name__ == '__main__':
    lstm = LSTMClassifier(2, 100, 50, 50, 0.5, 2, 2, torch.device('cpu'))

    #variable length sequences
    X0 = torch.LongTensor([1, 5, 8, 19, 43])
    X1 = torch.LongTensor([23,44,5,13,1,34,43])
    X = [X1, X0] #sequence needs to be sorted in descending order of length

    X_padded = nn.utils.rnn.pad_sequence(X, batch_first=True)
    # print(X_padded)

    #In case of error, change line in forward: embs = embeds.view(sentence.shape[1], sentence.shape[0], -1)
    fwd_out = lstm.forward(X_padded, [7,5])


    labels = torch.LongTensor([[1,0],[0,1]])

    loss = lstm.loss(fwd_out, labels)
    print(loss)