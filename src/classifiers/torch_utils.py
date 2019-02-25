from __future__ import print_function

from src.data_proc.corpus_utils import CorpusEncoder

import torch.nn as nn
import torch
from torch.nn.parameter import Parameter

from os.path import exists, join, realpath
from os import makedirs


class TorchUtils:

    @staticmethod
    def get_sort_unsort(lengths):
        _, sort = torch.sort(lengths, descending=True)
        _, unsort = sort.sort()
        return sort, unsort

    @staticmethod
    def save_model(corpus_encoder, state, fname_state, dir_state, dir_encoder, fname_encoder = 'corpus_encoder.json'):
        '''
        Save model state along with relevant architecture parameters as a state dictionary
        :param corpus_encoder: encoder for corpus
        :param state: state dictionary with relevant details (e.g. network arch, epoch, model states and optimizer states)
        :param fname_state: out file name
        :param dir_out: out directory
        '''
        if not exists(dir_state):
            makedirs(dir_state)

        if not exists(dir_encoder):
            makedirs(dir_encoder)

        # serialize encoder
        corpus_encoder.to_json(fname_encoder, dir_encoder)

        #serialize model state
        torch.save(state, realpath(join(dir_state, fname_state)))

    @staticmethod
    def load_model(fname_state, dir_state, dir_encoder, fname_encoder = 'corpus_encoder.json'):
        '''
        Load dictionary of model state and arch params
        :param fname_state: state file name to load
        :param dir_state: directory with filename
        '''
        if not exists(realpath(join(dir_state, fname_state))):
            raise FileNotFoundError("Model not found")

        if not exists(realpath(join(dir_encoder, fname_encoder))):
            raise FileNotFoundError("Encoder not found")

        # load encoder
        corpus_encoder = CorpusEncoder.from_json(fname_encoder, dir_state)

        #load model state
        state = torch.load(realpath(join(dir_state, fname_state)))

        return state, corpus_encoder

class EmbeddingMul(nn.Module):
    """This class implements a custom embedding module which registers a hook to save gradients.
    Note: this class accepts the arguments from the original pytorch module
    but only with values that have no effects, i.e set to False, None or -1.
    """

    def __init__(self, num_embeddings, embedding_dim, padding_idx=None,
                 max_norm=None, norm_type=2, scale_grad_by_freq=False,
                 sparse=False, _weight=None):

        # Declares the same arguments as the original pytorch implementation but only for backward compatibility.
        # Their values must be set to have no effects.
        # Checks if unsupported argument are used
        # ____________________________________________________________________
        if padding_idx != None:
            raise NotImplementedError("padding_idx must be None, not %s".format(padding_idx))
        if max_norm is not None:
            raise NotImplementedError("max_norm must be None, not %s".format(max_norm))
        if scale_grad_by_freq:
            raise NotImplementedError("scale_grad_by_freq must be False, not %s".format(scale_grad_by_freq))
        if sparse:
            raise NotImplementedError("sparse must be False, not %s".format(sparse))
        # ____________________________________________________________________

        super(EmbeddingMul, self).__init__()

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        if _weight is None:
            self.weight = Parameter(torch.Tensor(num_embeddings, embedding_dim))
            self.reset_parameters()
        else:
            assert list(_weight.shape) == [num_embeddings, embedding_dim], \
                'Shape of weight does not match num_embeddings and embedding_dim'
            self.weight = Parameter(_weight)

        self._requires_grad = True #False
        self._requires_emb_grad = False

    @property
    def requires_grad(self):
        return self._requires_grad

    @requires_grad.setter
    def requires_grad(self, value):
        self._requires_grad = value

    @property
    def requires_emb_grad(self):
        return self._requires_emb_grad

    @requires_emb_grad.setter
    def requires_emb_grad(self, value):
        self._requires_emb_grad = value

    def reset_parameters(self):
        self.weight.data.normal_(0, 1)

    def forward(self, input):
        """
        Args:
            - input: of shape (seq_len, batch_size).
        Returns:
            - result: of shape (seq_len, batch_size, emb_dim_size)
        """
        #Wrapping as parameter is important to convert it as leaf node
        embs = Parameter(self.to_embeddings(input).to(self.weight.device))

        if self.requires_emb_grad:
            # registers hook to track gradients of the embedded sequences
            embs.register_hook(self.save_grad)
            # embs.register_hook(print)

        return embs

    def save_grad(self, grad):
        #grad shape: seq_len * batch_size * emb_dim
        self.last_grad = grad

    def to_embeddings(self, input):
        # Returns a new tensor that doesn't share memory

        #index_select picks out the vectors corresponding to the words for every word in the input
        #the first view makes a single dimensional sequence of input words across all instances
        #the second view reshapes it to seq_len * batch_size * emb_dim. '+' concatenates additional dimension

        #The requires_grad parameter of result will mimic requires_grad parameter of self
        with torch.set_grad_enabled(self.requires_grad):
            result = torch.index_select(
                self.weight, 0, input.view(-1).long()).view(
                input.size()+(self.embedding_dim,))

        return result

    def __repr__(self):
        # return self.__class__.__name__ + "({})".format(self.num_embeddings)
        s = '{num_embeddings}, {embedding_dim}'
        if self.padding_idx is not None:
            s += ', padding_idx={padding_idx}'
        if self.max_norm is not None:
            s += ', max_norm={max_norm}'
        if self.norm_type != 2:
            s += ', norm_type={norm_type}'
        if self.scale_grad_by_freq is not False:
            s += ', scale_grad_by_freq={scale_grad_by_freq}'
        if self.sparse is not False:
            s += ', sparse=True'
        return s.format(**self.__dict__)


if __name__ == "__main__":
    input = torch.tensor([[1, 2, 0], [3, 4, 5]])
    emb_vocab = 10
    emb_dim = 5
    mod = EmbeddingMul(emb_vocab, emb_dim)
    print(mod.weight)
    print(mod.weight.shape)
    output = mod(input)
    print(output)
    print(output.shape)