from src.utils import FileUtils

import argparse
import pandas as pd
import numpy as np

from os.path import realpath, join
import statistics

PATH_MIMICIII = ''
FNAME_DIAGNOSES = 'DIAGNOSES_ICD.csv.gz'
FNAME_NOTES = 'NOTEEVENTS.csv.gz'

# ICD-9-CM codes for sepsis, severe sepsis and septic shock
ICD9_SEPSIS = "99591"
ICD9_SEVERE_SEPSIS = "99592"
ICD9_SEPTIC_SHOCK = "78552"

PATH_MIMICIII_SEPSIS = ''
PATH_MIMICIII_SEPSIS_TEXT = join(PATH_MIMICIII_SEPSIS, 'text')
PATH_MIMICIII_SEPSIS_LABELS = join(PATH_MIMICIII_SEPSIS, 'labels')
FNAME_LABELS = 'sepsis_labels.json'


class SepsisMIMIC:

    def get_septic(self, sepsis_codes):

        diag_df = PandasUtils.load_csv(FNAME_DIAGNOSES, PATH_MIMICIII)
        hadm_ids = self.select_septic_hadm_id(diag_df, sepsis_codes)
        sepsis_notes_df = self.get_septic_notes(hadm_ids)

    def select_septic_hadm_id(self, diag_df, sepsis_codes):
        print("Getting septic HADM_IDs")
        hadm_ids = diag_df[diag_df['ICD9_CODE'].isin(sepsis_codes)]['HADM_ID']
        # print("Septic HADM_IDs \n", list(hadm_ids))
        return list(hadm_ids)

    def get_septic_notes(self, septic_hadm_ids,
                         fname_notes=FNAME_NOTES, dir_in=PATH_MIMICIII):
        print("Loading notes csv")
        notes_df = PandasUtils.load_csv(fname_notes, dir_in)

        print("Removing error entries")
        prev_len = notes_df.shape[0]
        notes_df = notes_df[notes_df['ISERROR'] != 1]
        assert notes_df.shape[0] < prev_len, "None of the entries are removed"

        print("Removing leading and trailing spaces and converting text to lowercase")
        notes_df['TEXT'] = notes_df['TEXT'].str.strip()
        print("Converting text to lowercase")
        notes_df['TEXT'] = notes_df['TEXT'].str.lower()

        print("Removing blank and NA entries from TEXT and HADM_ID columns")
        notes_df['TEXT'].replace('', np.nan, inplace=True)
        notes_df.dropna(subset=['HADM_ID', 'TEXT'], inplace=True)

        print("Converting HADM ID to int")
        notes_df['HADM_ID'] = notes_df['HADM_ID'].astype('int64')
        print("Converting chartdate to datetime")
        notes_df['CHARTDATE'] = pd.to_datetime(notes_df['CHARTDATE'], format='%Y-%m-%d')
        # print("All data types", notes_df.dtypes)

        print("Dropping duplicates")
        notes_df = notes_df.drop_duplicates(subset=['SUBJECT_ID', 'HADM_ID',
                                                    'CHARTDATE', 'CHARTTIME',
                                                    'CATEGORY', 'DESCRIPTION',
                                                    'TEXT'],
                                            keep='first')

        print("Adding septic labels")
        notes_df['SEPTIC'] = np.where(notes_df['HADM_ID'].isin(septic_hadm_ids),
                                      "septic", "non_septic")

        len_all_notes = [len(cur_note.split()) for cur_note in list(notes_df['TEXT'])]
        print("Average length of notes: ", statistics.mean(len_all_notes))
        print("Total number of notes: ", len(len_all_notes))

        print("Number of septic notes: ", notes_df[notes_df['SEPTIC'] == "septic"].shape[0])

        print("All categories of notes")
        print(set(notes_df['CATEGORY']))

        print("Retaining only discharge notes")
        notes_df = notes_df[notes_df['CATEGORY'] == "Discharge summary"]

        print("New categories, ", set(notes_df['CATEGORY']))

        print("Total Number of notes: ", notes_df.shape[0])
        print("Number of septic notes: ", notes_df[notes_df['SEPTIC'] == "septic"].shape[0])

        note_subset = notes_df.loc[notes_df.groupby('HADM_ID').CHARTDATE.idxmax()]
        print("Number of notes after selecting last note per admission: ", note_subset.shape[0])
        print("Number of septic notes after selecting last note per admission: ",
              note_subset[note_subset['SEPTIC'] == "septic"].shape[0])

        print("Serializing data")
        label_dict = {}  # {"HADM_ID":"septic"|"non-septic"}
        for cur_hadm_id in note_subset['HADM_ID'].tolist():
            cur_label = note_subset[note_subset['HADM_ID'] == cur_hadm_id]['SEPTIC'].item()
            label_dict[str(cur_hadm_id)] = cur_label
            text = note_subset[note_subset['HADM_ID'] == cur_hadm_id]['TEXT'].item()
            FileUtils.write_txt(text, str(cur_hadm_id)+'.txt', PATH_MIMICIII_SEPSIS_TEXT)

        # write labels json file
        FileUtils.write_json(label_dict, FNAME_LABELS, PATH_MIMICIII_SEPSIS_LABELS)

        # pandas dataframe as csv?
        note_subset.to_csv(join(PATH_MIMICIII_SEPSIS, "mimic_sepsis_subset_df.csv"))


class PandasUtils:
    @staticmethod
    def load_csv(fname, dir_in, dtype=None):
        return pd.read_csv(realpath(join(dir_in, fname)), dtype=dtype)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--mimic_dir", dest='mimic_dir',
                        help="Path containing the MIMIC-III notes zipped files.",
                        required=True)

    parser.add_argument("--sepsis_out_dir", dest='sepsis_out_dir',
                        help="Path to write sepsis notes and labels to.",
                        required=True)

    args = parser.parse_args()

    PATH_MIMICIII = args.mimic_dir
    PATH_MIMICIII_SEPSIS = args.sepsis_out_dir

    sepsis_obj = SepsisMIMIC()
    sepsis_obj.get_septic([ICD9_SEPSIS, ICD9_SEVERE_SEPSIS, ICD9_SEPTIC_SHOCK])