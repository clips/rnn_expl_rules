import sys
sys.path.append('/home/madhumita/PycharmProjects/rnn_expl_rules/')

from src.corpus_utils import DataUtils, SST2Corpus, CorpusEncoder
from src.classifiers.lstm import LSTMClassifier
from src.explanations.grads import Explanation
from src.explanations.group_expl import ImpTopics
from src.explanations.expl_orig_input import SeqSkipGram
from src.weka_utils.vec_to_arff import get_feat_dict, write_arff_file

from os.path import exists, realpath, join
import resource

from spacy.lang.en import English

soft, hard = 5.4e+10, 5.4e+10  # nearly 50GB
resource.setrlimit(resource.RLIMIT_AS, (soft, hard))


PATH_DIR_CORPUS = '../dataset/sst2/'
FNAME_TRAIN = 'train_binary_sent.csv'
FNAME_VAL = 'dev_binary_sent.csv'
FNAME_TEST = 'test_binary_sent.csv'

FNAME_ENCODER = 'corpus_encoder_sentiment.json'
PATH_DIR_ENCODER = '../out/'

model_name = 'lstm'  # lstm|gru
test_mode = 'test'  # val | test

baseline = False


def init_spacy_eng_tokenizer():
    nlp = English()
    tokenizer = nlp.Defaults.create_tokenizer(nlp)
    return tokenizer


def load_model():
    classifier = LSTMClassifier.load(
        f_model='sentiment_lstm_classifier_hid150_emb300.tar')

    return classifier


def load_corpora():

    # initialize corpora
    tokenizer = init_spacy_eng_tokenizer()

    train_corp = SST2Corpus(FNAME_TRAIN, PATH_DIR_CORPUS, 'train', tokenizer)
    val_corp = SST2Corpus(FNAME_VAL, PATH_DIR_CORPUS, 'val', tokenizer)
    test_corp = SST2Corpus(FNAME_TEST, PATH_DIR_CORPUS, 'test', tokenizer)

    if not exists(realpath(join(PATH_DIR_ENCODER, FNAME_ENCODER))):
        raise FileNotFoundError("Encoder not found")
    # load encoder
    corpus_encoder = CorpusEncoder.from_json(FNAME_ENCODER, PATH_DIR_ENCODER)

    return train_corp, val_corp, test_corp, corpus_encoder


def get_imp_topics(eval_corp, classifier, encoder, get_imp=False):

    explanations = dict()
    method = 'dot'

    if get_imp:
        print("Computing word importance scores")
        explanation = Explanation.get_grad_importance(method,
                                                      classifier,
                                                      eval_corp,
                                                      encoder)
        explanations[method] = explanation
    else:
        print("Loading word importance scores")
        explanations[method] = Explanation.from_imp(method,
                                                    classifier,
                                                    eval_corp,
                                                    encoder)

    seqs = encoder.get_decoded_sequences(eval_corp)
    imp_topics = ImpTopics.from_seqs_scores(seqs, scores=explanations['dot'].imp_scores)
    imp_topics.get_topic_words()


def main():
    classifier = load_model()
    train_corpus, val_corpus, test_corpus, corpus_encoder = load_corpora()
    get_imp_topics(train_corpus, classifier, corpus_encoder, get_imp=True)



if __name__ == '__main__':
    main()

