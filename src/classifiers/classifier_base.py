from abc import ABCMeta, abstractmethod

from src.explanations.grads import Explanation

import torch.nn as nn
import torch
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score, accuracy_score
import numpy as np

from random import shuffle


class RNNClassifier(nn.Module):
    __metaclass__ = ABCMeta

    def __init__(self, batch_size):
        super().__init__()

        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        self.batch_size = batch_size  # this parameter is architecture independent

    @property
    @abstractmethod
    def hidden_in(self):
        pass

    @abstractmethod
    def forward(self, sentence, sent_lengths, hidden):
        pass

    @abstractmethod
    def save(self, f_model, dir_model):
        pass

    @abstractmethod
    def load(self, f_model, dir_model):
        pass

    def detach_hidden_(self):
        if self.hidden_in is None:
            return

        if type(self.hidden_in) == tuple:
            for node in self.hidden_in:
                node.detach_()
        else:
            self.hidden_in.detach_()

    def loss(self, fwd_out, target, weight_tensor):
        # NLL loss to be used when logits have log-softmax output.
        # If softmax layer is not added, directly CrossEntropyLoss can be used.
        loss_fn = nn.NLLLoss(weight=weight_tensor)
        return loss_fn(fwd_out, target)

    def train_model(self, corpus, corpus_encoder, n_epochs, optimizer, val_corpus=None,
                    weighted_loss=False):

        if weighted_loss:
            # IMP: the following would break if y does not contain all possible classes
            label_weights = compute_class_weight(class_weight='balanced',
                                                 classes=list(set(corpus.labels)),
                                                 y=corpus.labels)
            label_weights = torch.from_numpy(label_weights).type(torch.FloatTensor)
            label_weights = label_weights.to(self.device)
            print("Label weights: ", label_weights)
        else:
            label_weights = None

        optimizer = optimizer

        # initialize the early_stopping object
        early_stopping = EarlyStopping(is_lower_better=False, patience=5, verbose=True)

        for i in range(n_epochs):

            running_loss = 0.0

            # shuffle the corpus
            shuffle(corpus.row_ids)

            # get train batch
            for idx, (cur_insts, cur_labels) in enumerate(
                    corpus_encoder.get_batches_from_corpus(corpus, self.batch_size)):

                torch.cuda.empty_cache()
                
                cur_insts, cur_labels, cur_lengths = corpus_encoder.batch_to_tensors(
                                                        cur_insts, cur_labels, self.device)

                self.train()

                # forward pass
                fwd_out = self.forward(cur_insts, cur_lengths, self.hidden_in)

                # loss calculation
                loss = self.loss(fwd_out, cur_labels, weight_tensor=label_weights)

                # backprop
                optimizer.zero_grad()  # reset tensor gradients
                loss.backward()  # compute gradients for network params w.r.t loss

                # nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)

                optimizer.step()  # perform the gradient update step

                # detach hidden nodes from the graph.
                # IMP to prevent the graph from growing.
                self.detach_hidden_()

                # print statistics
                running_loss += loss.item()

                if idx % 100 == 99:  # print every 100 mini-batches
                    print('[%d, %5d] loss: %.3f' %
                          (i + 1, idx + 1, running_loss / 100))
                    running_loss = 0.0

            if val_corpus:
                y_pred_val, y_true_val = self.predict(val_corpus, corpus_encoder)
                y_score_val = f1_score(y_true=y_true_val, y_pred=y_pred_val,
                                       average='macro')
                print("Validation macro F1: ",
                      f1_score(y_true=y_true_val, y_pred=y_pred_val,
                               average='macro'))

                early_stopping(y_score_val, self)

                if early_stopping.early_stop:
                    print("Early stopping")
                    break

        # load the last checkpoint with the best model
        self = self.load('checkpoint.pt')

    def predict(self, corpus, corpus_encoder):

        self.eval()

        y_pred = list()
        y_true = list()

        for idx, (cur_insts, cur_labels) in enumerate(
                corpus_encoder.get_batches_from_corpus(corpus, self.batch_size)):
            cur_insts, cur_labels, cur_lengths = corpus_encoder.batch_to_tensors(
                                                    cur_insts, cur_labels, self.device)

            y_true.extend(cur_labels.cpu().numpy())

            self.detach_hidden_()

            # forward pass
            fwd_out = self.forward(cur_insts, cur_lengths, self.hidden_in)

            cur_preds = torch.argmax(fwd_out.detach(), 1)
            y_pred.extend(cur_preds.cpu().numpy())

        return y_pred, y_true

    def predict_from_insts(self, texts, encoder, get_prob=False):
        """
        :param texts: 2D list, n_inst * n_words for every instance
        :param encoder: corpus encoder object
        :param get_prob: True to get probability output
        :return: output prediction -- class/prob
        """
        self.eval()

        preds = list()

        for cur_batch in encoder.get_batches_from_insts(texts, self.batch_size):

            # tensors shape maxlen * n_inst
            # lengths is a list of lengths
            tensors, __, lengths = encoder.batch_to_tensors(cur_batch, None, self.device)

            self.detach_hidden_()

            # forward pass
            fwd_out = self.forward(tensors, lengths, self.hidden_in)

            if get_prob:
                fwd_out = torch.exp(fwd_out)  # back from log_softmax to softmax
                preds.extend(fwd_out.detach().cpu().numpy())
            else:
                cur_preds = torch.argmax(fwd_out.detach(), 1)
                preds.extend(cur_preds.cpu().numpy())

        return preds

    def get_importance(self, corpus, corpus_encoder, eval_obj):
        """
        Compute word importance scores based on backpropagated gradients
        """

        # methods = ['dot', 'sum', 'max', 'l2', 'max_mul', 'l2_mul', 'mod_dot']
        methods = ['dot']

        explanations = dict()

        for cur_method in methods:
            print("Pooling method: ", cur_method)

            explanation = Explanation.get_grad_importance(cur_method, self, corpus, corpus_encoder)
            explanations[cur_method] = explanation

            eval_obj.avg_prec_recall_f1_at_k_from_corpus(explanation.imp_scores, corpus, corpus_encoder, k=15)

        return explanations


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, is_lower_better, patience=5, verbose=False):
        """
        Args:
            is_lower_better (bool): If True, a lower value fo the val_metric is better.
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
        """
        self.is_lower_better = is_lower_better
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_metric_min = np.Inf

    def __call__(self, val_metric, model):

        if self.is_lower_better:
            score = -val_metric
        else:
            score = val_metric

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_metric, model)
        elif score < self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_metric, model)
            self.counter = 0

    def save_checkpoint(self, val_metric, model):
        """
        Saves model when validation loss decrease.
        """
        if self.verbose:
            print("Validation score changed from {} --> {}. Saving model ... ".format(
                self.val_metric_min, val_metric))
        model.save('checkpoint.pt')
        self.val_metric_min = val_metric
