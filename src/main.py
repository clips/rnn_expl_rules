import sys
sys.path.append('/home/madhumita/PycharmProjects/rnn_expl_rules/')

from src.corpus_utils import spacy_eng_tokenizer, dummy_processor
from src.utils import FileUtils
from src.train_classifiers import process_model
from src.get_explanation_files import get_explanation_files

from os.path import realpath
import resource
soft, hard = 5.4e+10, 5.4e+10  # nearly 50GB
resource.setrlimit(resource.RLIMIT_AS, (soft, hard))


class SepsisMIMIC:
    PATH_DIR_CORPUS = realpath('../dataset/sepsis_mimic/')
    FNAME_TRAIN = 'train_sepsis_mimic.csv'
    FNAME_VAL = 'val_sepsis_mimic.csv'
    FNAME_TEST = 'test_sepsis_mimic.csv'
    PATH_DIR_OUT = realpath('../out/')

    TOKENIZER = dummy_processor
    LABEL_DICT = {"non_septic": 0, "septic": 1}

    PRETRAINED_EMBS = False

    load_encoder = True
    FNAME_ENCODER = 'corpus_encoder_mimiciii.json'
    PATH_ENCODER = realpath('../out/')

    train_model = False
    model_name = 'lstm'
    n_layers = 2
    n_hid = 100
    n_emb = 100
    dropout = 0.
    bidir = True

    test_mode = 'test'  # val | test


class Newsgroups:
    PATH_DIR_CORPUS = realpath('../dataset/newsgroups/')
    print(PATH_DIR_CORPUS)
    FNAME_TRAIN = 'train_newsgroups.csv'
    FNAME_VAL = 'val_newsgroups.csv'
    FNAME_TEST = 'test_newsgroups.csv'
    FNAME_LABELDICT = 'newsgroups_labeldict.json'
    PATH_DIR_OUT = realpath('../out/')

    TOKENIZER = spacy_eng_tokenizer
    LABEL_DICT = FileUtils.read_json(FNAME_LABELDICT, PATH_DIR_CORPUS)

    PRETRAINED_EMBS = True
    PATH_DIR_EMBS = '/home/corpora/word_embeddings/'
    FNAME_EMBS = 'glove.840B.300d.txt'
    N_DIM_EMBS = 300
    embs_from_disk = True
    FNAME_EMBS_WT = 'pretrained_embs_newsgroups.npy'

    load_encoder = True
    FNAME_ENCODER = 'corpus_encoder_newsgroups.json'
    PATH_ENCODER = realpath('../out/')

    train_model = True
    model_name = 'lstm'
    n_layers = 3
    n_hid = 600
    n_emb = N_DIM_EMBS
    dropout = 0.3
    bidir = True

    test_mode = 'val'  # val | test


class SST2:
    PATH_DIR_CORPUS = realpath('../dataset/sst2/')
    FNAME_TRAIN = 'train_binary_sent.csv'
    FNAME_VAL = 'dev_binary_sent.csv'
    FNAME_TEST = 'test_binary_sent.csv'
    PATH_DIR_OUT = realpath('../out/')

    TOKENIZER = spacy_eng_tokenizer
    LABEL_DICT = {"positive": 1, "negative": 0}

    PRETRAINED_EMBS = True
    PATH_DIR_EMBS = '/home/corpora/word_embeddings/'
    FNAME_EMBS = 'glove.840B.300d.txt'
    N_DIM_EMBS = 300
    embs_from_disk = True
    FNAME_EMBS_WT = 'pretrained_embs_sentiment.npy'

    load_encoder = True
    FNAME_ENCODER = 'corpus_encoder_sentiment.json'
    PATH_ENCODER = realpath('../out/')

    train_model = False
    model_name = 'lstm'
    n_layers = 1
    n_hid = 150
    n_emb = N_DIM_EMBS
    dropout = 0.
    bidir = False

    test_mode = 'test'  # val | test


SUPPORTED_DATASETS = ('sepsis-mimic', 'sepsis-mimic-discharge',
                      'newsgroups', 'sst2')
dataset_class_dict = {'sepsis-mimic': SepsisMIMIC,
                      'newsgroups': Newsgroups,
                      'sst2': SST2
                      }


def main(dataset, get_expl, get_baseline_expl):

    if dataset.lower() not in SUPPORTED_DATASETS:
        raise ValueError("Please enter one of the following supported datasets: ",
                         SUPPORTED_DATASETS)
    dataset_class = dataset_class_dict[dataset.lower()]

    process_model(dataset_class)

    if get_expl:
        get_explanation_files(dataset_class, get_baseline_expl)


if __name__ == '__main__':
    # newsgroups | sst2 | sepsis-mimic | sepsis-mimic-discharge
    ds = 'newsgroups'
    # ds = 'sepsis-mimic'
    get_expl = False
    get_baseline_expl = False
    main(ds, get_expl, get_baseline_expl)
