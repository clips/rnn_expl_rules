from src.clamp.clamp_proc import Clamp
from src.utils import FileUtils

from os.path import join, realpath, exists
from os import makedirs
import argparse
import re
import gzip
import random
from itertools import chain


class SyntheticSepsis:

    def __init__(self):
        self.regex_pneumonia = r"pneumonia"
        self.regex_empyema = r"empyema"

        # the following three infection regex could be combined into one
        self.regex_meningitis = r"meningitis"
        self.regex_endocarditis = r"endocarditis"
        self.regex_other_infections = r"infection"

        self.regex_labs_temp = r"hyperthermia | hypothermia"
        self.regex_labs_wbc = r"leukocytosis | leukopenia"
        self.regex_mental_status = r"altered mental status"
        self.regex_labs_tachycardia = r"tachycardia"
        self.regex_labs_tachypnea = r"tachypnea"
        self.regex_labs_hyperglycemia = r"hyperglycemia"

        # number of docs to create based on the regex match
        self.n_sepsis_sample_docs = 50000

        # docs with only those sentences without a regex match
        self.n_negative_docs = 20000

    def get_sentence_shortlist(self, f_in, dir_in):

        self.pneumonia_and_empyema_sents = set()
        self.meningitis_sents = set()
        self.endocarditis_sents = set()
        self.other_infection_sents = set()

        self.temp_sents = set()
        self.wbc_sents = set()
        self.mental_status_sents = set()
        self.tachycardia_sents = set()
        self.tachypnea_sents = set()
        self.hyperglycemia_sents = set()

        self.negative_sents = set()

        with gzip.open(realpath(join(dir_in, f_in))) as mimic_corpus:
            for line in mimic_corpus:
                line = line.decode('utf-8').lower()

                # limiting to sentences with 3--15 words
                if len(line.split()) >= 15 or len(line.split()) < 3:
                    continue

                if self.is_pneumonia_and_empyema(line):
                    self.pneumonia_and_empyema_sents.add(line)

                elif self.get_regex_match(line, self.regex_meningitis):
                    self.meningitis_sents.add(line)

                elif self.get_regex_match(line, self.regex_endocarditis):
                    self.endocarditis_sents.add(line)

                elif self.get_regex_match(line, self.regex_other_infections):
                    self.other_infection_sents.add(line)

                elif self.get_regex_match(line, self.regex_labs_temp):
                    self.temp_sents.add(line)

                elif self.get_regex_match(line, self.regex_labs_wbc):
                    self.wbc_sents.add(line)

                elif self.get_regex_match(line, self.regex_mental_status):
                    self.mental_status_sents.add(line)

                elif self.get_regex_match(line, self.regex_labs_tachycardia):
                    self.tachycardia_sents.add(line)

                elif self.get_regex_match(line, self.regex_labs_tachypnea):
                    self.tachypnea_sents.add(line)

                elif self.get_regex_match(line, self.regex_labs_hyperglycemia):
                    self.hyperglycemia_sents.add(line)

                elif len(self.negative_sents) < (self.n_sepsis_sample_docs +
                                                 self.n_negative_docs) * 10:  # 10 random sentences per document
                    self.negative_sents.add(line)

        print("Number of pneumonia, empyema sentences: ",
              len(self.pneumonia_and_empyema_sents))
        print("Number of meningitis sentences: ",
              len(self.meningitis_sents))
        print("Number of endocarditis sentences: ",
              len(self.endocarditis_sents))
        print("Number of other infection sentences: ",
              len(self.other_infection_sents))

        print("Number of temperature measurement sentences:", len(self.temp_sents))
        print("Number of WBC measurement sentences: ", len(self.wbc_sents))
        print("Number of mental status sentences: ", len(self.mental_status_sents))
        print("Number of tachycardia sentences: ", len(self.tachycardia_sents))
        print("Number of tachypnea sentences: ", len(self.tachypnea_sents))
        print("Number of hyperglycemia sentences: ", len(self.hyperglycemia_sents))
        print("Number of negative sentences: ", len(self.negative_sents))

    def is_pneumonia_and_empyema(self, str):
        return (re.findall(self.regex_pneumonia, str) and
                re.findall(self.regex_empyema, str))

    @staticmethod
    def get_regex_match(str, regex):
        return re.findall(regex, str)

    def create_docs(self, dir_out):

        if not exists(dir_out):
            makedirs(dir_out)

        for i in range(self.n_sepsis_sample_docs + self.n_negative_docs):

            if i < self.n_sepsis_sample_docs:
                # populating rnn_expl_rules sample docs
                sent_list = self._get_septic_sentence_shortlist()
            else:
                # populating negative docs
                sent_list = list()

                # only negative sentences
                sent_list.extend(random.sample(self.negative_sents, 17))

            random.shuffle(sent_list)

            with open(realpath(join(dir_out, str(i) + '.txt')), 'w') as f_out:
                for line in sent_list:
                    f_out.write(line)

    def _get_septic_sentence_shortlist(self):

        sent_list = list()

        # one sentence mentioning an infection
        infection_sents = set(chain(self.pneumonia_and_empyema_sents,
                                    self.meningitis_sents,
                                    self.endocarditis_sents,
                                    self.other_infection_sents))
        sent_list.extend(random.sample(infection_sents, 1))

        # one sentence about the physical condition of the patient
        sent_list.extend(random.sample(self.temp_sents, 1))
        sent_list.extend(random.sample(self.wbc_sents, 1))
        sent_list.extend(random.sample(self.mental_status_sents, 1))
        sent_list.extend(random.sample(self.tachycardia_sents, 1))
        sent_list.extend(random.sample(self.tachypnea_sents, 1))
        sent_list.extend(random.sample(self.hyperglycemia_sents, 1))

        # 10 sentences that do not match any of the keywords
        sent_list.extend(random.sample(self.negative_sents, 10))

        return sent_list

    def get_septic_labels(self, dir_clamp, dir_labels):

        clamp_obj = Clamp()

        labels = dict()
        n_septic = 0

        for i in range(self.n_sepsis_sample_docs + self.n_negative_docs):
            cur_label = 'non_septic'

            is_infected = False
            n_present_labs = 0

            entities = clamp_obj.get_entities(str(i) + '.txt', dir_clamp)

            for cur_entity in entities:
                if not is_infected:
                    if (self.is_pneumonia_and_empyema(cur_entity.mention) or
                        self.get_regex_match(cur_entity.mention,
                                             self.regex_meningitis) or
                        self.get_regex_match(cur_entity.mention,
                                             self.regex_endocarditis) or
                        self.get_regex_match(cur_entity.mention,
                                             self.regex_other_infections)
                    ) and cur_entity.assertion.lower() == 'present':
                        # infection term is mentioned and not negated in the sentence
                        is_infected = True

                if n_present_labs < 2:
                    if (self.get_regex_match(cur_entity.mention,
                                             self.regex_labs_temp) or
                        self.get_regex_match(cur_entity.mention,
                                             self.regex_labs_wbc) or
                        self.get_regex_match(cur_entity.mention,
                                             self.regex_mental_status) or
                        self.get_regex_match(cur_entity.mention,
                                             self.regex_labs_tachycardia) or
                        self.get_regex_match(cur_entity.mention,
                                             self.regex_labs_tachypnea) or
                        self.get_regex_match(cur_entity.mention,
                                             self.regex_labs_hyperglycemia)
                    ) and cur_entity.assertion.lower() == 'present':
                        # patient condition term is mentioned and
                        # not negated in the sentence
                        n_present_labs += 1

            # septic if the patient has an infection and
            # at least two of the pre-specified conditions
            if is_infected and n_present_labs >= 2:
                cur_label = 'septic'
                n_septic += 1

            labels[str(i)] = cur_label

        print("Number of instances labeled as septic: {} of total {} instances"
              .format(n_septic, self.n_sepsis_sample_docs + self.n_negative_docs))

        FileUtils.write_json(labels, 'sepsis_labels.json', dir_labels)
        return labels


def create_synthetic_text(f_mimic, dir_mimic, dir_out):

    synthetic_sepsis = SyntheticSepsis()

    # populate documents
    synthetic_sepsis.get_sentence_shortlist(f_mimic, dir_mimic)
    synthetic_sepsis.create_docs(dir_out)

    return synthetic_sepsis


def add_labels(template, dir_clamp, dir_labels):
    template.get_septic_labels(dir_clamp, dir_labels)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--dir_mimic", dest='dir_mimic',
                        help="Path containing the tokenized MIMIC-III notes.",
                        required=True)

    parser.add_argument("--dir_out", dest='dir_out',
                        help="Path to write the text files of the synthetic data.",
                        required=True)

    parser.add_argument("--dir_clamp", dest='dir_clamp',
                        help="Path containing clamp output files of the synthetic data text.",
                        required=True)

    parser.add_argument("--dir_labels", dest='dir_clamp',
                        help="Path containing the labels for the synthetic texts.",
                        required=True)

    args = parser.parse_args()

    dir_mimic = args.dir_mimic
    dir_out = args.dir_out
    dir_clamp = args.dir_clamp
    dir_labels = args.dir_labels

    f_mimic = 'mimic-notes-tokenized.txt.gz'

    synthetic_dataset = create_synthetic_text(f_mimic, dir_mimic, dir_out)


    # IMP!!!
    # First run clamp on all the documents before running the next command
    add_labels(synthetic_dataset, dir_clamp, dir_labels)

    # @todo: make a shell pipeline to create text, run clamp, obtain labels, ...
    # @todo: ... tokenize text, and then write the dataset as a csv
