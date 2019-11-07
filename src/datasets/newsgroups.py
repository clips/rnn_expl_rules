from src.utils import FileUtils

from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import StratifiedShuffleSplit

from os.path import realpath, join
from os import path, makedirs
from itertools import zip_longest
import csv


def get_train_test_split(x, y, test_ratio=0.1, seed=0):
    """
    Stratified split of training data_proc into training and validation sets
    :param x: original training feats
    :param y: original training labels
    :return: new train feats, test feats, train labels, and test labels
    """
    sss = StratifiedShuffleSplit(n_splits = 1, test_size = test_ratio,
                                 random_state = seed)
    for train_idx, val_idx in sss.split(x, y):
        x_train, y_train = [x[i] for i in train_idx.tolist()], y[train_idx]
        x_val, y_val = [x[i] for i in val_idx.tolist()], y[val_idx]

    return x_train, x_val, y_train, y_val


def write_csv(texts, labels, fname, dir_name, header=["label", "text"]):

    if not path.exists(dir_name):
        makedirs(dir_name)

    with open(realpath(join(dir_name, fname)), 'w') as f:
        csv_writer = csv.writer(f)

        if header is not None:
            csv_writer.writerow(header)

        for cur_label, cur_text in zip_longest(labels, texts):
            if cur_text.strip() == '':
                continue
            csv_writer.writerow([cur_label, cur_text])


class Newsgroups:

    def __init__(self, cats, dir_csv):
        """
        :param cats: List of categories to limit to
        """
        self.cats = cats
        self.x_train, self.y_train = None, None
        self.x_val, self.y_val = None, None
        self.x_test, self.y_test = None, None

        self.label_dict = None

        self.get_data()
        self.write_dataset(dir_csv)

    def get_data(self):
        """
        Gathers and prepares the 20 newsgroups dataset after removing
        headers, footers and quotes.
        :return train, val, test set as list of text and labels
        """

        remove_cont = ('headers', 'footers', 'quotes')

        ds_train = fetch_20newsgroups(subset='train',
                                      remove=remove_cont,
                                      categories=self.cats)
        ds_test = fetch_20newsgroups(subset='test',
                                     remove=remove_cont,
                                     categories=self.cats)

        self.x_train, self.x_val, self.y_train, self.y_val = get_train_test_split(
            ds_train.data, ds_train.target)

        self.x_test, self.y_test = ds_test.data, ds_test.target

        self.label_dict = {i:label for i, label in enumerate(ds_train.target_names)}

    def write_dataset(self, dir_out):
        write_csv(self.x_train, self.y_train, 'train_newsgroups.csv', dir_out)
        write_csv(self.x_val, self.y_val, 'val_newsgroups.csv', dir_out)
        write_csv(self.x_test, self.y_test, 'test_newsgroups.csv', dir_out)

        FileUtils.write_json(self.label_dict, 'newsgroups_labeldict.json', dir_out)


if __name__ == "__main__":
    Newsgroups(cats=None, dir_csv='../../dataset/newsgroups/')

