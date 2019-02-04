from src.utils import FileUtils

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

import torch
from os.path import realpath, join
# from random import shuffle

#beginning of seq, end of seq, beg of line, end of line, unknown, padding symbol
BOS, EOS, BOL, EOL, UNK, PAD = '<s>', '</s>', '<bol>', '</bol>', '<unk>', '<pad>'

class Vocab:
    def __init__(self):
        self.word2idx = dict() #word to index lookup
        self.idx2word = dict() #index to word lookup

        self.reserved_sym = dict() #dictionary of reserved terms with corresponding symbols.

    @classmethod
    def populate_indices(cls, vocab_set, **reserved_sym):
        inst = cls()

        for key, sym in reserved_sym.items(): #populate reserved symbols such as bos, eos, unk, pad
            if sym in vocab_set:
                print("Removing the reserved symbol {} from training corpus".format(sym))
                del vocab_set[sym] #@todo: delete symbol from embedding space also
            inst.word2idx.setdefault(sym, len(inst.word2idx))  # Add item with given default value if it does not exist.
            inst.reserved_sym[key] = sym # Populate dictionary of reserved symbols. @todo: check data type of key. Var?
            setattr(cls, key, inst.word2idx[sym]) # Add reserved symbols as class attributes with corresponding idx mapping

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

def dummy_processor(line):
    return line.strip().split()

def encode_labels(fit_labels, transform_labels):
    le = LabelEncoder()
    le.fit(fit_labels)
    print("Label classes: ", list(le.classes_), "respectively mapped to ", le.transform(le.classes_))
    return le.transform(transform_labels)

class Corpus:
    def __init__(self, dir_corpus, f_labels, dir_labels, fname_subset, text_processor = dummy_processor, label_encoder = encode_labels):
        self.dir_in = dir_corpus
        self.fname_subset = fname_subset  # file names for the current split of the corpus

        all_labels = FileUtils.read_json(f_labels, dir_labels)
        self.labels = [all_labels[i] for i in self.fname_subset]
        all_labels = list(all_labels.values())
        self.labels = label_encoder(all_labels, self.labels)

        self.text_processor = text_processor

    def __iter__(self):
        # if is_shuffle:
        #     combined = list(zip(self.fname_subset, self.labels))
        #     shuffle(combined)
        #     cur_split, cur_labels = zip(*combined)
        # else:
        #     cur_split, cur_labels = self.fname_subset, self.labels

        for cur_fname, cur_label in zip(self.fname_subset, self.labels):
            with open(realpath(join(self.dir_in, cur_fname + '.txt'))) as f:
                word_seq = list()
                for line in f:
                    word_seq.extend(self.text_processor(line))
                yield (word_seq, cur_label)

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
                    if not word in vocab_set:
                        vocab_set.add(word)

        # create vocabs
        vocab = Vocab.populate_indices(vocab_set, bos=BOS, eos=EOS, bol=BOL, eol=EOL, unk=UNK, pad=PAD)
        return cls(vocab)

    def encode_inst(self, inst):
        # inst = self.transform(inst)
        # encoded_inst = [self.vocab.get(i) for i in inst] #@todo: redundant because transform_item already replaces with idx. confirm.
        # return encoded_inst
        '''
        Converts sentence to sequence of indices after adding beginning, end and replacing unk tokens.
        @todo: check if beg and end of seq and line are required for our classification setup.
        '''
        out = [self.transform_item(i) for i in inst]
        if self.vocab.bos is not None:
            out = [self.vocab.bos] + out
        if self.vocab.eos is not None:
            out = out + [self.vocab.eos]
        return out

    # def transform(self, inst):


    def transform_item(self, item):
        '''
        Returns the index for an item if present in vocab, <unk> otherwise.
        '''
        try:
            return self.vocab.word2idx[item]
        except KeyError:
            if self.vocab.unk is None:
                raise ValueError("Couldn't retrieve <unk> for unknown token")
            else:
                return self.vocab.unk

    def get_batches(self, corpus, batch_size):

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

    def batch_to_tensors(self, cur_insts, cur_labels, device):
        '''
        Transforms an encoded batch to the corresponding torch tensor
        :return: tensor of batch padded to maxlen, and a tensor of actual instance lengths
        '''
        lengths = [len(inst) for inst in cur_insts]
        n_inst, maxlen = len(cur_insts), max(lengths)

        t = torch.zeros(n_inst, maxlen, dtype=torch.int64) + self.vocab.pad #this creates a tensor of padding indices

        #copy the sequence
        for idx, (inst, length) in enumerate(zip(cur_insts, lengths)):
            t[idx, :length].copy_(torch.tensor(inst))

        #contiguous() makes a copy of tensor so the order of elements would be same as if created from scratch.
        t = t.t().contiguous().to(device)
        lengths = torch.tensor(lengths).to(device)
        labels = torch.LongTensor(cur_labels).to(device)

        return t, labels, lengths


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

        return (train_split, val_split, test_split)

    @staticmethod
    def create_splits(doc_ids, labels):
        train_idx, rest_idx, __, rest_labels = train_test_split(doc_ids, labels, stratify=labels, test_size=0.2)
        val_idx, test_idx = train_test_split(rest_idx, stratify=rest_labels, test_size=0.5)

        return (train_idx, val_idx, test_idx)

    @staticmethod
    def read_splits(dir_splits):
        train_split = FileUtils.read_list('train_ids.txt', dir_splits)
        val_split = FileUtils.read_list('val_ids.txt', dir_splits)
        test_split = FileUtils.read_list('test_ids.txt', dir_splits)

        return (train_split, val_split, test_split)