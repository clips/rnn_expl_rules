from __future__ import print_function

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
    def save_model(state, fname_state, dir_state):
        '''
        Save model state along with relevant architecture parameters as a state dictionary
        :param state: state dictionary with relevant details (e.g. network arch, epoch, model states and optimizer states)
        :param fname_state: out file name
        :param dir_state: out directory
        '''
        if not exists(dir_state):
            makedirs(dir_state)

        #serialize model state
        torch.save(state, realpath(join(dir_state, fname_state)))

    @staticmethod
    def load_model(fname_state, dir_state):
        '''
        Load dictionary of model state and arch params
        :param fname_state: state file name to load
        :param dir_state: directory with filename
        '''
        if not exists(realpath(join(dir_state, fname_state))):
            raise FileNotFoundError("Model not found")

        #load model state
        state = torch.load(realpath(join(dir_state, fname_state)))

        return state



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
        # if padding_idx != None:
        #     raise NotImplementedError("padding_idx must be None, not %s".format(padding_idx))
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

        if padding_idx is not None:
            if padding_idx > 0:
                assert padding_idx < self.num_embeddings, 'Padding_idx must be within num_embeddings'
            elif padding_idx < 0:
                assert padding_idx >= -self.num_embeddings, 'Padding_idx must be within num_embeddings'
                padding_idx = self.num_embeddings + padding_idx
        self.padding_idx = padding_idx

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
        if self.padding_idx is not None:
            self.weight.data[self.padding_idx].fill_(0)

    def forward(self, input):
        """
        Args:
            - input: of shape (seq_len, batch_size).
        Returns:
            - result: of shape (seq_len, batch_size, emb_dim_size)
        """
        #Wrapping as parameter is important to convert it as leaf node
        # if not self.requires_emb_grad:
        embs = Parameter(self.to_embeddings(input).to(self.weight.device))
        # else:
        #     embs = Parameter(self.product_for_embedding(input).to(self.weight.device))

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

    # def product_for_embedding(self, input):
    #     # The requires_grad parameter of result will mimic requires_grad parameter of self
    #     self.last_oh = self.to_one_hot(input)
    #
    #     with torch.set_grad_enabled(self.requires_grad):
    #         result = torch.stack(
    #             [torch.mm(batch.float(), self.weight.cpu()) for batch in self.last_oh], dim=0)
    #     return result
    #
    # def to_one_hot(self, input):
    #     # Returns a new tensor that doesn't share memory
    #
    #     ones = torch.eye(self.num_embeddings, requires_grad=False)
    #     result = torch.index_select(
    #         ones, 0, input.cpu().view(-1).long()).view(
    #         input.size() + (self.num_embeddings,))
    #
    #     result.requires_grad = self.requires_grad
    #
    #     return result

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

    @classmethod
    def from_pretrained(cls, embeddings, freeze=True):
        r"""Creates Embedding instance from given 2-dimensional FloatTensor.

        Args:
            embeddings (Tensor): FloatTensor containing weights for the Embedding.
                First dimension is being passed to Embedding as 'num_embeddings', second as 'embedding_dim'.
            freeze (boolean, optional): If ``True``, the tensor does not get updated in the learning process.
                Equivalent to ``embedding.weight.requires_grad = False``. Default: ``True``
        """
        assert embeddings.dim() == 2, \
            'Embeddings parameter is expected to be 2-dimensional'
        rows, cols = embeddings.shape
        embedding = cls(num_embeddings=rows, embedding_dim=cols, _weight=embeddings)
        embedding.weight.requires_grad = not freeze
        return embedding


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