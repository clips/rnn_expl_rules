from src.utils import FileUtils

import warnings
import torch
import numpy as np
from os.path import splitext


class Explanation:
    """
    Base class for all explanations.
    Parameters: model, data, technique, explanation
    """
    def __init__(self, method, model, corpus, encoder, imp_scores, preds):
        self.method = method
        self.model = model
        self.corpus = corpus
        self.corpus_encoder = encoder
        self.preds = preds  #@todo: shall we move this to model?
        self.imp_scores = imp_scores

    @classmethod
    def get_grad_importance(cls, grad_pooling, model, corpus, corpus_encoder):
        """
        Compute word importance scores based on backpropagated gradients
        :param grad_pooling: (dot|sum|max|l2|max_mul|l2_mul)
                              pooling technique for combining embedding dimension importance into word importance
        :param model: model to compute importance scores for
        :param corpus: corpus to explain
        :param corpus_encoder: encoder used for the given corpus
        """
        grad_pooling = grad_pooling.lower()

        if grad_pooling not in {'dot', 'sum', 'max', 'l2', 'max_mul', 'l2_mul', 'mod_dot'}:
            raise ValueError("Enter a supported pooling technique (dot|sum|max|l2|max_mul|l2_mul|mod_dot)")

        model.eval()
        # IMP! backward doesnt work in eval mode unless we disable cudnn
        torch.backends.cudnn.enabled = False  # disabling cudnn allows for backward in eval mode

        # activating setting to register hook. Needs to be done before the forward pass.
        model.word_embeddings.requires_emb_grad = True

        global_imp_lst = list()
        pred_lst = list()

        for idx, (cur_insts, cur_labels) in enumerate(corpus_encoder.get_batches_from_corpus(corpus, model.batch_size)):
            cur_insts, cur_labels, cur_lengths = corpus_encoder.batch_to_tensors(cur_insts, cur_labels, model.device)

            # forward pass
            fwd_out = model.forward(cur_insts, cur_lengths, model.hidden_in)
            preds = torch.argmax(fwd_out.detach(), 1)

            # converting log softmax to softmax for gradient computation
            fwd_out = torch.exp(fwd_out)

            # sequence embeddings. Shape seq_len * batch_size * emb_dim
            embs = model.word_embeddings(cur_insts)

            # create tensor to specify the nodes to compute the gradients of
            grad_tensor = torch.zeros_like(fwd_out)
            grad_tensor[np.arange(len(preds)), preds] = 1

            model.zero_grad()

            fwd_out.backward(grad_tensor)

            # Get the grads saved in the hook here, shape: seq_len * batch_size * emb_dim
            grads = model.word_embeddings.last_grad

            model.detach_hidden_() # free up the computation graph

            # word_imp shape: batch_size * seq_len
            word_imp = getattr(GradPooling, grad_pooling)(grads, embs)
            # keeping the importance of valid timesteps only
            for row, cols in enumerate(cur_lengths):
                global_imp_lst.append(word_imp[row, :cols].tolist())
            pred_lst.extend(preds.tolist())

        inst = cls('grad_' + grad_pooling, model, corpus, corpus_encoder,
                   global_imp_lst,
                   pred_lst)

        inst.save(fname='imp_scores_' +
                        model.model_type +
                        '_hid' + str(model.hidden_dim) +
                        '_emb' + str(model.emb_dim) +
                        '_' + splitext(corpus.fname)[0] +
                        '_' + grad_pooling + '.json'
                  )
        return inst

    def save(self, fname, dir_out='../out/'):

        seqs = self.corpus_encoder.get_decoded_sequences(self.corpus,
                                                         strip_angular=True)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            golds = self.corpus.label_encoder.inverse_transform(self.corpus.get_labels())
            preds = self.corpus.label_encoder.inverse_transform(self.preds)

            if not isinstance(golds, list): golds = golds.tolist()
            if not isinstance(preds, list): preds = preds.tolist()

        # saving the sequences, the importance scores, and the gold and predicted labels as JSON file

        FileUtils.write_json(
            {'seq_lst': seqs,
             'imp_scores': self.imp_scores,
             'gold': golds,
             'pred': preds},
            fname, dir_out)

    @classmethod
    def from_imp(cls, pooling, model, corpus, encoder, dir_in='../out/'):

        fname = 'imp_scores_' + model.model_type + \
                '_hid' + str(model.hidden_dim) + '_emb' + str(model.emb_dim) + \
                '_' + splitext(corpus.fname)[0] + '_' + pooling + '.json'

        json_file = FileUtils.read_json(fname, dir_in)

        inst = cls(pooling, model, corpus, encoder,
                   json_file['imp_scores'],
                   corpus.label_encoder.transform(json_file['pred']))
        return inst


class GradPooling:
    @staticmethod
    def dot(grads, embs):
        # Take dot product between grads and emb_weights to get overall word imp scores.
        # Would be the same as taking element-wise product and them summing across emb_dim
        return torch.mul(embs, grads).sum(dim = 2).detach().transpose(0, 1)

    @staticmethod
    def sum(grads, embs = None):
        # total importance of all the dimensions of a word embedding to get overall importance
        return torch.sum(grads, dim=2).detach().transpose(0, 1)

    @staticmethod
    def l2(grads, embs = None):
        # Square the magnitude of gradients and sum across all the dimensions of a word
        return torch.pow(grads, 2).sum(dim=2).detach().transpose(0, 1)

    @staticmethod
    def max(grads, embs = None):
        # max the importance of all the dimensions of a word embedding to get overall importance
        return torch.max(grads, dim=2)[0].detach().transpose(0, 1)

    @staticmethod
    def max_mul(grads, embs):
        # max of element-wise product
        return torch.mul(embs, grads).max(dim=2)[0].detach().transpose(0, 1)

    @staticmethod
    def l2_mul(grads, embs):
        # l2 of element-wise product
        return torch.pow(torch.mul(embs, grads), 2).sum(dim=2).detach().transpose(0, 1)

    @staticmethod
    def mod_dot(grads, embs):
        # use absolute embedding values for product and retain sign of gradient, then sum across dimensions
        return torch.mul(abs(embs), grads).sum(dim=2).detach().transpose(0, 1)
