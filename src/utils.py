from os.path import exists, join, realpath
from os import makedirs
import json
from collections import OrderedDict, Counter

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
