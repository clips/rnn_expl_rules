import sys
sys.path.append('/home/madhumita/PycharmProjects/rnn_expl_rules/')

from src.utils import FileUtils

from sklearn.model_selection import train_test_split

import argparse
import csv
from os.path import realpath, join
from os import path, makedirs


def split_data(fname_lst, label_lst):

    train_idx, rest_idx, __, rest_labels = train_test_split(fname_lst, label_lst,
                                                            stratify=label_lst,
                                                            test_size=0.2)
    val_idx, test_idx = train_test_split(rest_idx, stratify=rest_labels, test_size=0.5)

    return train_idx, val_idx, test_idx


# def read_splits(dir_splits):
#     train_split = FileUtils.read_list('train_ids.txt', dir_splits)
#     val_split = FileUtils.read_list('val_ids.txt', dir_splits)
#     test_split = FileUtils.read_list('test_ids.txt', dir_splits)
#     return train_split, val_split, test_split


def write_csv(corpus_fidx, label_dict, dir_corpus, dir_clamp,
              fname_csv, dir_csv, header=None):

    if not path.exists(dir_csv):
        makedirs(dir_csv)

    if header and dir_clamp:
        header.append("clamp")

    with open(realpath(join(dir_csv, fname_csv)), 'w') as f_csv:

        csv_writer = csv.writer(f_csv)

        if header:
            csv_writer.writerow(header)

        for cur_fname in corpus_fidx:
            with open(realpath(join(dir_corpus, cur_fname + '.txt'))) as f:
                text = f.read()
                label = 1 if label_dict[cur_fname] == "septic" else 0

            if dir_clamp is None:
                csv_writer.writerow([label, text])
            else:
                with open(realpath(join(dir_clamp, cur_fname + '.txt'))) as f:
                    clamp = f.read()
                csv_writer.writerow([label, text, clamp])


def main(f_labels, dir_labels, dir_corpus, fname_suffix, dir_csv, dir_clamp=None):

    label_dict = FileUtils.read_json(f_labels, dir_labels)
    sorted_labels = sorted(label_dict.items())
    fname_lst = [i for i, j in sorted_labels]  # file names in sorted order
    all_labels_lst = [j for i, j in sorted_labels]  # labels sorted according to file names
    train_idx, val_idx, test_idx = split_data(fname_lst, all_labels_lst)
    # train_idx, val_idx, test_idx = read_splits('/home/madhumita/sepsis_synthetic/splits/')

    write_csv(train_idx, label_dict, dir_corpus, dir_clamp,
              'train_'+fname_suffix+'.csv', dir_csv, ["label", "text"])

    write_csv(val_idx, label_dict, dir_corpus, dir_clamp,
              'val_'+fname_suffix+'.csv', dir_csv, ["label", "text"])

    write_csv(test_idx, label_dict, dir_corpus, dir_clamp,
              'test_'+fname_suffix+'.csv', dir_csv, ["label", "text"])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--dir_sepsis_synthetic", dest='dir_sepsis_synthetic',
                        help="Directory containing relevant synthetic dataset files.",
                        required=True)

    parser.add_argument("--dir_sepsis_mimic", dest='dir_sepsis_mimic',
                        help="Directory containing relevant sepsis-mimic dataset files.",
                        required=True)

    parser.add_argument("--dir_sepsis_mimic_discharge", dest='dir_sepsis_mimic_discharge',
                        help="Directory containing relevant sepsis-mimic-discharge dataset files.",
                        required=True)

    args = parser.parse_args()

    dir_synthetic = args.dir_sepsis_synthetic
    dir_mimic = args.dir_sepsis_mimic
    dir_mimic_discharge = args.dir_sepsis_mimic_discharge

    f_labels = 'sepsis_labels.json'
    dir_labels = join(dir_synthetic, 'labels')
    dir_corpus = join(dir_synthetic, 'text')
    dir_clamp = join(dir_synthetic, 'clamp')
    fname_suffix = 'synthetic'
    dir_csv = '../../../dataset/sepsis_synthetic/'
    main(f_labels, dir_labels, dir_corpus, fname_suffix, dir_csv, dir_clamp)

    f_labels = 'sepsis_labels.json'
    dir_labels = join(dir_mimic, 'labels')
    dir_corpus = join(dir_mimic, 'text')
    fname_suffix = 'sepsis_mimic'
    dir_csv = '../../../dataset/sepsis_mimic/'
    main(f_labels, dir_labels, dir_corpus, fname_suffix, dir_csv)

    f_labels = 'sepsis_labels.json'
    dir_labels = join(dir_mimic_discharge, 'labels')
    dir_corpus = join(dir_mimic_discharge, 'text')
    fname_suffix = 'sepsis_mimic_discharge'
    dir_csv = '../../../dataset/sepsis_mimic_discharge/'
    main(f_labels, dir_labels, dir_corpus, fname_suffix, dir_csv)



