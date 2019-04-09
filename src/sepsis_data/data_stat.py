from os.path import join, realpath
from collections import Counter
import random
import math
import string

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt


def type_token_ratio(fname, dir_name, no_punc = True, sample_n = None, n_runs = 10):
    '''
    Computes type token ratio for a given corpus
    If sample_n is not None, it samples sample_n tokens from the corpus and averages the TTR scores over several samples.
    :param fname: corpus name
    :param dir_name: corpus directory
    :param no_punc: True to remove punctuation
    :param sample_n: number of tokens to sample, None otherwise
    :param n_runs: number of runs to averaged sampled TTR scores over
    :return: TTR score
    '''

    with open(realpath(join(dir_name, fname))) as f:
        tokens = f.read().split()

        if no_punc:
            tokens = [i for i in tokens if i not in string.punctuation]

        if sample_n:
            ttrs = set()
            for i in range(n_runs):
                ttr = get_ttr(random.sample(tokens, sample_n))
                ttrs.add(ttr)
            avg_ttr = sum(ttrs) / len(ttrs)
            print("Averaged ttr over {} runs: {}".format(n_runs, avg_ttr))
            return avg_ttr

        else:
            ttr = get_ttr(tokens)
            print("Type token ratio:", ttr)
            return ttr

def get_ttr(tokens):
    types = set(tokens)
    # print("Num of tokens: ", len(tokens))
    # print("Num of types: ", len(types))
    ttr = len(types)/len(tokens)
    return ttr



def zipfian_distr(fname, dir_name, dir_out, plot_name, corpus_name):
    '''
    Plot zipfian distribution of given corpus
    :param fname: corpus name
    :param dir_name: corpus directory
    :param dir_out: output directory for plot
    :param plot_name: plot name
    :param corpus_name: name of corpus
    '''
    with open(realpath(join(dir_name, fname))) as f:
        tokens = f.read().split()
        token_freq = Counter(tokens)

        tokens = token_freq.most_common(20000)

        rank = [i for i in range(len(tokens))]
        freq = [math.log(freq) for word, freq in tokens]

        plt.plot(rank, freq, label = corpus_name)
        plt.xlabel("Rank")
        plt.ylabel("Log frequency")

        plt.legend()

        plt.savefig(realpath(join(dir_out, plot_name)), format = 'png')


def plot_distr(fname_synthetic, dir_synthetic, fname_mimic, dir_mimic, dir_out, plot_name = 'zipf.png'):
    zipfian_distr(fname_synthetic, dir_synthetic, dir_out, plot_name, 'synthetic')
    zipfian_distr(fname_mimic, dir_mimic, dir_out, plot_name, 'MIMIC-III')


if __name__ == "__main__":
    print("Synthetic dataset TTR")
    type_token_ratio('concatenated.txt', '/home/madhumita/sepsis_synthetic/')

    print("MIMIC corpus, downsampled to size of synthetic corpus, TTR")
    type_token_ratio('mimic-notes-tokenized.txt', '/home/madhumita/', sample_n=11351153)

    print("Synthetic dataset, MSTTR over 100 token samples")
    type_token_ratio('concatenated.txt', '/home/madhumita/sepsis_synthetic/', sample_n= 100, n_runs = 110000)

    print("MIMIC corpus, downsampled to size of synthetic corpus, MSTTR over 100 token samples")
    type_token_ratio('mimic-notes-tokenized.txt', '/home/madhumita/', sample_n=100, n_runs = 110000)

    plot_distr('concatenated.txt', '/home/madhumita/sepsis_synthetic/',
               'mimic-notes-tokenized.txt', '/home/madhumita/', '../../out/')