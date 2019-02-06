from src.data_proc.corpus_utils import CorpusEncoder

import torch
from os.path import exists, join, realpath, splitext
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
        corpus_encoder.to_json(dir_encoder, fname_encoder)

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
        corpus_encoder = CorpusEncoder.from_json(dir_state, fname_encoder)

        #load model state
        state = torch.load(realpath(join(dir_state, fname_state)))

        return state, corpus_encoder