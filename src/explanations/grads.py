from src.torch_utils import TorchUtils
from src.utils import FileUtils

import warnings
import torch
import numpy as np

class Explanation:
    '''
    Base class for all explanations.
    Parameters: model, data, and explanation?
    '''
    def __init__(self, method):
        self.method = method
        # save explanations in attribute imp_scores later

    @classmethod
    def get_grad_importance(cls, model, corpus, corpus_encoder, grad_pooling, model_type):
        '''
        Compute word importance scores based on backpropagated gradients
        :param model: model to compute importance scores for
        :param corpus: corpus to explain
        :param corpus_encoder: encoder used for the given corpus
        :param grad_pooling: (dot|sum|max|l2|max_mul|l2_mul)
                              pooling technique for combining embedding dimension importance into word importance
        :param model_type: gru/lstm
        '''
        grad_pooling = grad_pooling.lower()

        if grad_pooling not in {'dot', 'sum', 'max', 'l2', 'max_mul', 'l2_mul', 'mod_dot'}:
            raise ValueError("Enter a supported pooling technique (dot|sum|max|l2|max_mul|l2_mul|mod_dot)")

        inst = cls('grad_' + grad_pooling)

        #activating setting to register hook. Needs to be done before the forward pass.
        model.word_embeddings.requires_emb_grad = True

        global_imp_lst = list()
        gold_lst = list()
        pred_lst = list()

        # model.eval() # IMP! backward doesnt work in eval mode - open issue.
        # Setting model to train mode and then toggling parameters manually
        model.train()
        TorchUtils._set_eval(model)

        for idx, (cur_insts, cur_labels) in enumerate(corpus_encoder.get_batches(corpus, model.batch_size)):
            cur_insts, cur_labels, cur_lengths = corpus_encoder.batch_to_tensors(cur_insts, cur_labels, model.device)

            # forward pass
            fwd_out = model.forward(cur_insts, cur_lengths, model.hidden_in)
            preds = torch.argmax(fwd_out.detach(), 1)

            #converting log softmax to softmax for gradient computation
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

            model.detach_hidden_() #free up the computation graph

            # word_imp shape: batch_size * seq_len
            word_imp = getattr(GradPooling, grad_pooling)(grads, embs)
            # keeping the importance of valid timesteps only
            for row, cols in enumerate(cur_lengths):
                global_imp_lst.append(word_imp[row, :cols].tolist())
            # global_imp_lst.extend(word_imp.tolist())

            # recording gold and predicted labels for saving the JSON file later
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                cur_labels = corpus.label_encoder.inverse_transform(cur_labels.tolist())
                preds = corpus.label_encoder.inverse_transform(preds.tolist())
            gold_lst.extend(cur_labels)
            pred_lst.extend(preds)

        inst.imp_scores = global_imp_lst

        seq_lst = corpus_encoder.get_decoded_sequences(corpus, strip_angular = True)

        inst.save(global_imp_lst, seq_lst, pred_lst, gold_lst,
                  fname = 'imp_scores_' +
                          model_type +
                          '_hid' + str(model.hidden_dim) +
                          '_emb' + str(model.emb_dim) +
                          '_' + grad_pooling + '.json'
                 )
        return inst

    def save(self, imp_scores, seqs, preds, golds, fname, dir_out = '../out/'):
        # saving the sequences, the importance scores, and the gold and predicted labels as JSON file
        FileUtils.write_json(
            {'seq_lst': seqs,
             'imp_scores': imp_scores,
             'gold': golds,
             'pred': preds},
            fname, dir_out)


class GradPooling:
    @staticmethod
    def dot(grads, embs):
        # Take dot product between grads and emb_weights to get overall word imp scores.
        # Would be the same as taking element-wise product and them summing across emb_dim
        return torch.mul(embs, grads).sum(dim = 2).detach().transpose(0,1)

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