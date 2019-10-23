from src.utils import FileUtils

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

import torch
from os.path import realpath, join
import json
import csv
import numpy as np
from random import shuffle

# beginning of seq, end of seq, beg of line, end of line, unknown, padding symbol
BOS, EOS, BOL, EOL, UNK, PAD = '<s>', '</s>', '<bol>', '</bol>', '<unk>', '<pad>'


class Vocab:
    def __init__(self):
        self.word2idx = dict()  # word to index lookup
        self.idx2word = dict()  # index to word lookup

        self.reserved_sym = dict()  # dictionary of reserved terms with corresponding symbols.

    @classmethod
    def populate_indices(cls, vocab_set, **reserved_sym):
        inst = cls()

        # populate reserved symbols such as bos, eos, unk, pad
        for key, sym in reserved_sym.items():
            if sym in vocab_set:
                print("Removing the reserved symbol {} from training corpus".format(sym))
                del vocab_set[sym]
            # Add item with given default value if it does not exist.
            inst.word2idx.setdefault(sym, len(inst.word2idx))
            inst.reserved_sym[key] = sym  # Populate dictionary of reserved symbols.
            # Add reserved symbols as class attributes with corresponding idx mapping
            setattr(cls, key, inst.word2idx[sym])

        for term in vocab_set:
            inst.word2idx.setdefault(term, len(inst.word2idx))

        inst.idx2word = {val: key for key, val in inst.word2idx.items()}

        return inst

    def __getitem__(self, item):
        return self.word2idx[item]

    def __len__(self):
        return len(self.word2idx)

    @property
    def size(self):
        return len(self.word2idx)

    def to_dict(self):
        return {"reserved": self.reserved_sym,
                'word2idx': [{"key": key, "val": val} for key, val in self.word2idx.items()]}

    @classmethod
    def from_dict(cls, d):
        inst = cls()
        # the paramter "d" here is the return value of to_dict function earlier.
        inst.word2idx = {d["key"]: d["val"] for d in d['word2idx']}
        for key, val in d['reserved'].items():
            setattr(inst, key, inst.word2idx[val])
        inst.idx2word = {val: key for key, val in inst.word2idx.items()}

        return inst


def dummy_processor(line, lower=True):
    if lower:
        line = line.lower()
    return line.strip().split()


def encode_labels(fit_labels, transform_labels):
    le = LabelEncoder()
    le.fit(fit_labels)
    print("Label classes: ", list(le.classes_),
          "respectively mapped to ", le.transform(le.classes_))
    return le.transform(transform_labels), le


class Corpus:
    def __init__(self, dir_corpus, f_labels, dir_labels, fname_subset, subset_name,
                 text_processor=dummy_processor, label_encoder=encode_labels,
                 resample=False):
        self.dir_in = dir_corpus
        self.subset_ids = fname_subset  # file names for the current split of the corpus
        self.subset_name = subset_name

        all_labels = FileUtils.read_json(f_labels, dir_labels)
        self.labels = [all_labels[i] for i in self.subset_ids]
        self.labels, self.label_encoder = label_encoder(list(all_labels.values()),
                                                        self.labels)

        if resample:
            self.subset_ids = DataUtils.downsample(self.subset_ids,
                                                   self.labels)
            resampled_labels = [all_labels[i] for i in self.subset_ids]
            self.labels = self.label_encoder.transform(resampled_labels)

        self.text_processor = text_processor

    def __iter__(self):

        for cur_fname, cur_label in zip(self.subset_ids, self.labels):
            with open(realpath(join(self.dir_in, cur_fname + '.txt'))) as f:
                word_seq = list()
                for line in f:
                    word_seq.extend(self.text_processor(line))
                yield (word_seq, cur_label)

    def get_class_distribution(self):
        for cur_label in set(self.labels):
            print("Percentage of instances for class{}: {}".
                  format(cur_label, sum(self.labels==cur_label)/len(self.labels)*100))


class TorchNLPCorpus:
    def __init__(self, torchnlp_dataset, subset_name, all_labels,
                 label_encoder=encode_labels):
        self.dataset = torchnlp_dataset
        self.subset_name = subset_name
        self.subset_ids = [i for i in range(len(self.dataset.examples))]

        labels = [cur_inst.label[0] for cur_inst in self.dataset.examples]
        self.labels, self.label_encoder = label_encoder(all_labels, labels)

    def __iter__(self):

        for i, inst in enumerate(self.dataset.examples):
            yield (inst.text, self.labels[i])

    def get_class_distribution(self):
        for cur_label in set(self.labels):
            print("Percentage of instances for class{}: {}".
                  format(cur_label, sum(self.labels==cur_label)/len(self.labels)*100))


class SST2Corpus:
    def __init__(self, fname_corpus, dir_in, subset_name, text_processor):
        self.fname_corpus = fname_corpus
        self.dir_in = dir_in
        self.subset_name = subset_name
        self.text_processor = text_processor

        self.init_subset_ids_labels()

    def init_subset_ids_labels(self):
        self.subset_ids = list()
        self.labels = list()

        with open(realpath(join(self.dir_in, self.fname_corpus))) as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            for i, row in enumerate(reader):
                if i == 0:
                    continue
                self.subset_ids.append(i-1)
                self.labels.append(int(row[0]))

    def __iter__(self):

        with open(realpath(join(self.dir_in, self.fname_corpus))) as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            for i, row in enumerate(reader):
                if i == 0:
                    continue
                else:
                    tokens = [str(i) for i in self.text_processor(row[1].lower())]
                    label = int(row[0])
                    yield(tokens, label)

    def get_class_distribution(self):
        for cur_label in set(self.labels):
            print("Percentage of instances for class{}: {}".
                  format(cur_label, sum(self.labels==cur_label)/len(self.labels)*100))

class CorpusEncoder:

    def __init__(self, vocab):
        self.vocab = vocab

    @classmethod
    def from_corpus(cls, *corpora):
        # create vocab set for initializing Vocab class
        vocab_set = set()

        for corpus in corpora:
            for (words, labels) in corpus:
                for word in words:
                    if word not in vocab_set:
                        vocab_set.add(word)

        # create vocabs
        # @todo: add min and max freq to vocab items
        vocab = Vocab.populate_indices(vocab_set, unk=UNK, pad=PAD)#bos=BOS, eos=EOS, bol=BOL, eol=EOL),

        return cls(vocab)

    def encode_inst(self, inst):
        """
        Converts sentence to sequence of indices after adding beginning,
        end and replacing unk tokens.
        """
        out = [self.transform_item(i) for i in inst]
        # if self.vocab.bos is not None:
        #     out = [self.vocab.bos] + out
        # if self.vocab.eos is not None:
        #     out = out + [self.vocab.eos]
        return out

    def transform_item(self, item):
        """
        Returns the index for an item if present in vocab, <unk> otherwise.
        """
        try:
            return self.vocab.word2idx[item]
        except KeyError:
            if self.vocab.unk is None:
                raise ValueError("Couldn't retrieve <unk> for unknown token")
            else:
                return self.vocab.unk

    def get_batches_from_corpus(self, corpus, batch_size):

        instances = list()
        labels = list()

        for (cur_inst, cur_label) in iter(corpus):
            cur_inst = self.encode_inst(cur_inst)
            instances.append(cur_inst)
            labels.append(cur_label)
            if len(instances) == batch_size:
                yield (instances, labels)
                instances = list()
                labels = list()

        if instances:
            yield (instances, labels)

    def get_batches_from_insts(self, insts, batch_size):

        instances = list()

        for cur_inst in insts:
            cur_inst = self.encode_inst(cur_inst)
            instances.append(cur_inst)
            if len(instances) == batch_size:
                yield instances
                instances = list()

        if instances:
            yield instances

    def batch_to_tensors(self, cur_insts, cur_labels, device):
        """
        Transforms an encoded batch to the corresponding torch tensor
        :return: tensor of batch padded to maxlen, and a tensor of actual instance lengths
        """
        lengths = [len(inst) for inst in cur_insts]
        n_inst, maxlen = len(cur_insts), max(lengths)

        # this creates a tensor of padding indices
        t = torch.zeros(n_inst, maxlen, dtype=torch.int64) + self.vocab.pad

        # copy the sequence
        for idx, (inst, length) in enumerate(zip(cur_insts, lengths)):
            t[idx, :length].copy_(torch.tensor(inst))

        # contiguous() makes a copy of tensor so the order of elements would be
        # same as if created from scratch.
        t = t.t().contiguous().to(device)
        lengths = torch.tensor(lengths, dtype = torch.int).to(device)

        if cur_labels:
            labels = torch.LongTensor(cur_labels).to(device)
        else:
            labels = None

        return t, labels, lengths

    def decode_inst(self, inst):
        out = [self.vocab.idx2word[i] for i in inst if i != self.vocab.pad]
        return out

    def replace_unk(self, inst):
        out = [self.vocab.idx2word[self.vocab.unk]
               if i not in self.vocab.word2idx else i
               for i in inst]
        return out

    def get_decoded_sequences(self, corpus, strip_angular=False):
        instances = list()

        for (cur_inst, __) in iter(corpus):
            cur_inst = self.replace_unk(cur_inst)
            if strip_angular:
                # stripping angular brackets to support HTML rendering
                cur_inst = [i.strip('<>') for i in cur_inst]
            instances.append(cur_inst)

        return instances

    def to_json(self, fname, dir_out):
        with open(realpath(join(dir_out, fname)), 'w') as f:
            json.dump({'vocab': self.vocab.to_dict()}, f)

    @classmethod
    def from_json(cls, fname, dir_out):
        with open(realpath(join(dir_out, fname))) as f:
            obj = json.load(f)

        vocab = Vocab.from_dict(obj['vocab'])

        return cls(vocab)


class DataUtils:

    @staticmethod
    def split_data(f_labels, dir_labels, dir_splits):
        labels = FileUtils.read_json(f_labels, dir_labels)
        sorted_labels = sorted(labels.items())
        f_idx = [i for i, j in sorted_labels]  # file names in sorted order
        label_list = [j for i, j in sorted_labels]  # labels sorted according to file names

        train_split, val_split, test_split = DataUtils.create_splits(f_idx, label_list)

        FileUtils.write_list(train_split, 'train_ids.txt', dir_splits)
        FileUtils.write_list(val_split, 'val_ids.txt', dir_splits)
        FileUtils.write_list(test_split, 'test_ids.txt', dir_splits)

        return train_split, val_split, test_split

    @staticmethod
    def create_splits(doc_ids, labels):
        train_idx, rest_idx, __, rest_labels = train_test_split(doc_ids, labels,
                                                                stratify=labels,
                                                                test_size=0.2)
        val_idx, test_idx = train_test_split(rest_idx, stratify=rest_labels, test_size=0.5)

        return train_idx, val_idx, test_idx

    @staticmethod
    def read_splits(dir_splits):
        train_split = FileUtils.read_list('train_ids.txt', dir_splits)
        val_split = FileUtils.read_list('val_ids.txt', dir_splits)
        test_split = FileUtils.read_list('test_ids.txt', dir_splits)

        return train_split, val_split, test_split

    @staticmethod
    def downsample(idx, labels):
        """
        Downsamples the instances of all classes to the length of the minority class
        :param idx: list of filename indices for all instances
        :param labels: list of class labels for all instances, mapping idx
        :return: new list of filename indices with all classes downsampled to minority class
        """

        print("Original length: ", len(idx))

        inst_idx = dict()  # Indicies of each class' observations
        n_insts = dict()  # Number of observations in each class

        for cur_class in set(labels):
            inst_idx[cur_class] = [idx[i] for i in np.where(labels == cur_class)[0]]
            n_insts[cur_class] = len(inst_idx[cur_class])

        # find class with min samples
        min_class = min(n_insts, key=n_insts.get)
        print("Retaining {} samples of each class".format(n_insts[min_class]))

        new_idx = list()
        # For every observation of min len class,
        # randomly sample from other classes without replacement
        for cur_class in inst_idx.keys():
            downsampled = np.random.choice(inst_idx[cur_class],
                                           size=n_insts[min_class],
                                           replace=False)
            new_idx.extend(downsampled)

        # Shuffle sampled indices
        shuffle(new_idx)

        # print("New file subsets: ", new_idx)
        print("New length: ", len(new_idx))

        return new_idx
