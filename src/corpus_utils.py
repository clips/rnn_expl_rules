from sklearn.preprocessing import LabelEncoder
from spacy.lang.en import English
import spacy
import pandas as pd

import torch
from os.path import realpath, join
import json
# import numpy as np
# from random import shuffle

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


def dummy_processor():
    return space_tokenizer  # to make it compatible with spacy style call


def space_tokenizer(line):
    return line.strip().split()


def spacy_eng_tokenizer():
    nlp = English()
    tokenizer = nlp.Defaults.create_tokenizer(nlp)
    return tokenizer


def sklearn_label_encoder(fit_labels, transform_labels):
    le = LabelEncoder()
    le.fit(fit_labels)
    print("Label classes: ", list(le.classes_),
          "respectively mapped to ", le.transform(le.classes_))
    return le.transform(transform_labels), le


class DictLabelEncoder:
    def __init__(self, label_dict):
        self.label2idx = label_dict
        self.idx2label = dict()
        for label in label_dict.keys():
            self.idx2label[self.label2idx[label]] = label

    def transform(self, labels):
        transformed_labels = [self.label2idx[label] for label in labels]
        return transformed_labels

    def inverse_transform(self, labels):
        transformed_labels = [self.idx2label[label] for label in labels]
        return transformed_labels


# class SepsisMimicCorpus:
#     def __init__(self, dir_corpus, f_labels, dir_labels, fname_subset, subset_name,
#                  text_processor=dummy_processor, label_encoder=sklearn_label_encoder,
#                  resample=False):
#         self.dir_in = dir_corpus
#         self.subset_ids = fname_subset  # file names for the current split of the corpus
#         self.subset_name = subset_name
#
#         all_labels = FileUtils.read_json(f_labels, dir_labels)
#         self.labels = [all_labels[i] for i in self.subset_ids]
#         self.labels, self.label_encoder = label_encoder(list(all_labels.values()),
#                                                         self.labels)
#
#         if resample:
#             self.subset_ids = DataUtils.downsample(self.subset_ids,
#                                                    self.labels)
#             resampled_labels = [all_labels[i] for i in self.subset_ids]
#             self.labels = self.label_encoder.transform(resampled_labels)
#
#         self.text_processor = text_processor
#
#     def __iter__(self):
#
#         for cur_fname, cur_label in zip(self.subset_ids, self.labels):
#             with open(realpath(join(self.dir_in, cur_fname + '.txt'))) as f:
#                 word_seq = list()
#                 for line in f:
#                     word_seq.extend(self.text_processor(line))
#                 yield (word_seq, cur_label)
#
#     def get_class_distribution(self):
#         for cur_label in set(self.labels):
#             print("Percentage of instances for class{}: {}".
#                   format(cur_label, sum(self.labels==cur_label)/len(self.labels)*100))

class TorchNLPCorpus:
    def __init__(self, torchnlp_dataset, subset_name, all_labels,
                 label_encoder=sklearn_label_encoder):
        self.dataset = torchnlp_dataset
        self.subset_name = subset_name
        self.subset_ids = [i for i in range(len(self.dataset.examples))]

        labels = [cur_inst.label[0] for cur_inst in self.dataset.examples]
        self.labels, self.label_encoder = label_encoder(all_labels, labels)

    def __iter__(self):

        for i, inst in enumerate(self.dataset.examples):
            yield (inst.text, self.labels[i])


class CSVCorpus:
    def __init__(self, fname_corpus, dir_in, subset_name,
                 text_processor, label_dict, lower=True, le=DictLabelEncoder):
        self.fname = fname_corpus
        self.dir_in = dir_in
        self.subset_name = subset_name

        # the following ids are required to shuffle training set.
        n_rows = 0
        for chunk in pd.read_csv(open(realpath(join(self.dir_in, self.fname))),
                                 chunksize=64, encoding='utf-8'):
            n_rows += len(chunk)
        self.row_ids = [i+1 for i in range(n_rows)]  # +1 accounts for header.

        self.text_processor = text_processor()
        if isinstance(self.text_processor, spacy.tokenizer.Tokenizer):
            self.uses_spacy = True
        else:
            self.uses_spacy = False

        self.lower = lower
        self.label_encoder = le(label_dict)

    def __iter__(self):
        for cur_row_id in self.row_ids:
            df = pd.read_csv(realpath(join(self.dir_in, self.fname)),
                             usecols=['text', 'label'],
                             skiprows=range(1, cur_row_id), nrows=1)

            if self.lower:
                text = df['text'].iloc[0].lower()
            else:
                text = df['text'].iloc[0]

            text = self.text_processor(text)

            # if spacy tokenizer, convert spacy token types to string
            if self.uses_spacy:
                text = [cur_token.text for cur_token in text]

            yield(text, df['label'].iloc[0])

    def get_labels(self):
        labels = list()
        for cur_row_id in self.row_ids:
            df = pd.read_csv(realpath(join(self.dir_in, self.fname)),
                             usecols=['label'],
                             skiprows=range(1, cur_row_id), nrows=1)

            labels.append(df['label'].iloc[0])
        return labels


class ClampedCSVCorpus(CSVCorpus):

    def __init__(self, fname_corpus, dir_in, subset_name,
                 text_processor, label_dict, lower=True, le=DictLabelEncoder):

        super(ClampedCSVCorpus, self).__init__(fname_corpus, dir_in,
                                               subset_name,
                                               text_processor, label_dict, lower, le)

    def __iter__(self):
        yield from super(ClampedCSVCorpus, self).__iter__()

    def get_text_clamptxt(self):
        for chunk in pd.read_csv(realpath(join(self.dir_in, self.fname)),
                                 usecols=['clamp', 'text'],
                                 chunksize=1):
            yield (chunk['text'].iloc[0], chunk['clamp'].iloc[0])


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

    # @staticmethod
    # def downsample(idx, labels):
    #     """
    #     Downsamples the instances of all classes to the length of the minority class
    #     :param idx: list of filename indices for all instances
    #     :param labels: list of class labels for all instances, mapping idx
    #     :return: new list of filename indices with all classes downsampled to minority class
    #     """
    #
    #     print("Original length: ", len(idx))
    #
    #     inst_idx = dict()  # Indicies of each class' observations
    #     n_insts = dict()  # Number of observations in each class
    #
    #     for cur_class in set(labels):
    #         inst_idx[cur_class] = [idx[i] for i in np.where(labels == cur_class)[0]]
    #         n_insts[cur_class] = len(inst_idx[cur_class])
    #
    #     # find class with min samples
    #     min_class = min(n_insts, key=n_insts.get)
    #     print("Retaining {} samples of each class".format(n_insts[min_class]))
    #
    #     new_idx = list()
    #     # For every observation of min len class,
    #     # randomly sample from other classes without replacement
    #     for cur_class in inst_idx.keys():
    #         downsampled = np.random.choice(inst_idx[cur_class],
    #                                        size=n_insts[min_class],
    #                                        replace=False)
    #         new_idx.extend(downsampled)
    #
    #     # Shuffle sampled indices
    #     shuffle(new_idx)
    #
    #     # print("New file subsets: ", new_idx)
    #     print("New length: ", len(new_idx))
    #
    #     return new_idx

    @staticmethod
    def get_class_distribution(labels):
        for cur_label in set(labels):
            print("Percentage of instances for class{}: {}".
                  format(cur_label, labels.count(cur_label)/len(labels)*100))

