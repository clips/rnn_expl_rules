from src.torch_utils import TorchUtils
from src.utils import FileUtils

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

        # self.eval() # IMP! backward doesnt work in eval mode - open issue.
        TorchUtils._set_eval(model)
        #activating setting to register hook. Needs to be done before the forward pass.
        model.word_embeddings.requires_emb_grad = True

        # global_seq_lst = list()
        json_seq_lst = list()
        global_imp_lst = list()

        for idx, (cur_insts, cur_labels) in enumerate(corpus_encoder.get_batches(corpus, model.batch_size)):
            cur_insts, cur_labels, cur_lengths = corpus_encoder.batch_to_tensors(cur_insts, cur_labels, model.device)

            # forward pass
            fwd_out = model.forward(cur_insts, cur_lengths)
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
            if grad_pooling == 'dot':
                # Take dot product between grads and emb_weights to get overall word imp scores.
                # Would be the same as taking element-wise product and them summing across emb_dim
                word_imp = torch.mul(embs, grads).sum(dim = 2).detach().transpose(0,1)
            elif grad_pooling == 'mod_dot':
                # use absolute embedding values for product and retain sign of gradient
                word_imp = torch.mul(abs(embs), grads).sum(dim=2).detach().transpose(0, 1)
            elif grad_pooling == 'sum':
                # total importance of all the dimensions of a word embedding to get overall importance
                word_imp = torch.sum(grads, dim=2).detach().transpose(0, 1)
            elif grad_pooling == 'l2':
                # Square the magnitude of gradients and sum across all the dimensions of a word
                word_imp = torch.pow(grads, 2).sum(dim=2).detach().transpose(0,1)
            elif grad_pooling == 'max':
                # max the importance of all the dimensions of a word embedding to get overall importance
                word_imp = torch.max(grads, dim=2)[0].detach().transpose(0, 1)
            elif grad_pooling == 'max_mul':
                #max of element-wise product
                word_imp = torch.mul(embs, grads).max(dim=2)[0].detach().transpose(0, 1)
            elif grad_pooling == 'l2_mul':
                # l2 of element-wise product
                word_imp = torch.pow(torch.mul(embs, grads),2).sum(dim=2).detach().transpose(0, 1)

            #@todo: move this block to another function/class, does not fit here
            for cur_inst, cur_label, cur_pred in zip(cur_insts.transpose(0,1).tolist(), cur_labels, preds):
                cur_inst = corpus_encoder.decode_inst(cur_inst)
                #stripping angular brackets to support heatmap visualization
                cur_inst = [i.strip('<>') for i in cur_inst]

                #@todo: very hacky way to show labels in heatmaps. Update later.
                cur_label = 'GOLD_SEPTIC' if cur_label else 'GOLD_NONSEPTIC'
                cur_pred = 'PRED_SEPTIC' if cur_pred else 'PRED_NONSEPTIC'

                cur_inst[0] = cur_label + "," + cur_pred + ':' + cur_inst[0]
                json_seq_lst.append(cur_inst)

            global_imp_lst.extend(word_imp.tolist())

        FileUtils.write_json(
            {'article_lst': json_seq_lst,
             'decoded_lst': json_seq_lst,
             'attn_dists': global_imp_lst,
             'p_gens': global_imp_lst},
            'imp_scores_'+model_type+str(model.emb_dim)+'_'+grad_pooling+'.json', '../out/')

        inst.imp_scores = global_imp_lst
        return inst


