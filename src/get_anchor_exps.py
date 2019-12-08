import sys
sys.path.append('/home/madhumita/PycharmProjects/rnn_expl_rules/')

from src.main import SUPPORTED_DATASETS, dataset_class_dict
from src.explanations.anchor_text import get_anchor_exps

if __name__ == '__main__':

    # newsgroups | sst2 | sepsis-mimic | sepsis-mimic-discharge
    dataset = 'sst2'
    dataset = dataset.lower()

    if dataset not in SUPPORTED_DATASETS:
        raise ValueError("Please enter one of the following supported datasets: ",
                         SUPPORTED_DATASETS)

    if dataset not in ['sst2', 'newsgroups']:
        space_tokenizer = True
    else:
        space_tokenizer = False

    dataset_class = dataset_class_dict[dataset]

    get_anchor_exps(dataset_class, space_tokenizer)
