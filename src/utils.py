from os.path import exists, join, realpath
from os import makedirs
import json
from collections import OrderedDict, Counter

import numpy as np


class FileUtils:
    @staticmethod
    def write_json(obj_dict, fname, dir_out):

        if not exists(dir_out):
            makedirs(dir_out)

        with open(realpath(join(dir_out, fname)), 'w') as f:
            json.dump(obj_dict, f)

    @staticmethod
    def read_json(fname, dir_in):
        with open(realpath(join(dir_in, fname))) as f:
            return json.load(f)

    @staticmethod
    def write_list(data_list, fname, dir_out):

        if not exists(dir_out):
            makedirs(dir_out)

        with open(realpath(join(dir_out, fname)), 'w') as f:
            for term in data_list:
                f.write(term+'\n')

    @staticmethod
    def read_list(fname, dir_in):

        data = list()
        with open(realpath(join(dir_in, fname))) as f:
            for line in f:
                data.append(line.strip())

        return data

    @staticmethod
    def write_txt(string, fname, dir_out):

        if not exists(dir_out):
            makedirs(dir_out)

        with open(realpath(join(dir_out, fname)), 'w') as f:
            f.write(string)

    @staticmethod
    def write_numpy(array, fname, dir_in):
        fpath = realpath(join(dir_in, fname))
        np.save(fpath, array)

    @staticmethod
    def read_numpy(fname, dir_in):
        fpath = realpath(join(dir_in, fname))
        return np.load(fpath)


def get_top_items_dict(data_dict, k, order=False):
    """Get top k items in the dictionary by score.
    Returns a dictionary or an `OrderedDict` if `order` is true.
    """
    top = sorted(data_dict.items(), key=lambda x: x[1], reverse=True)[:k]
    if order:
        return OrderedDict(top)
    return dict(top)


def get_most_freq_items(items, k):
    """
    :param items: 2D list of items
    :param k: number of items to retain
    :return: set of most frequent items
    """
    # count the number of instances the term occurs in
    term2freq = Counter(x for xs in items for x in set(xs))
    reduced_set = dict(term2freq.most_common(k)).keys()  # frequency filter.
    return reduced_set


class EmbeddingUtils:

    @staticmethod
    def load_glove_model(fname, dir_in, emb_dim):
        print("Loading word vectors ...")
        with open(realpath(join(dir_in, fname)), encoding='utf8') as f_glove:
            model = dict()
            for line in f_glove:
                line = line.strip().split()
                word = line[0]

                try:
                    embedding = np.array([float(i) for i in line[1:]])
                    if len(embedding) != emb_dim:
                        continue
                    model[word] = embedding
                except:
                    continue
                    # print("Skipping incorrect entry in embedding file")

            return model

    @staticmethod
    def get_embedding_weight(fname, dir_in, emb_dim, vocab_dict):
        """
        :param emb_dict: Dictionary {term: vector array}
        :param vocab_dict: Dictionary {term: vocab ID}
        :return: 2D array of vectors n_terms * n_emb_dim
        """
        emb_dict = EmbeddingUtils.load_glove_model(fname, dir_in, emb_dim)

        unk_vec = np.mean(list(emb_dict.values()), axis=0)

        weights = np.full((len(vocab_dict), emb_dim), unk_vec)
        print("Weights matrix initialized. Shape: ", weights.shape)

        print("Updating weight matrix ... ")
        for word in vocab_dict.keys():
            if word in emb_dict:
                weights[vocab_dict[word]] = emb_dict[word]

        return weights
