import pandas as pd
from os.path import join, realpath

import numpy as np

import ucto

class SepsisData:

    def __init__(self, dir_sepsis, dir_mimic):

        self.read_mimic_notes(dir_mimic)
        sepsis_df = self.read_sepsis3_csv(dir_sepsis)
        self.label_sepsis(sepsis_df)

        mimic_shortlisted_df = self.keyword_filter()
        # self.analyze_notes(mimic_shortlisted_df)

    def read_sepsis3_csv(self, dir_in):

        fname = 'sepsis3-df.csv'

        print("Reading sepsis3 file")
        return pd.read_csv(realpath(join(dir_in, fname)))

    def read_mimic_notes(self, dir_in):

        fname = 'NOTEEVENTS.csv.gz'

        print("Reading MIMIC notes file")
        self.mimic_df = pd.read_csv(realpath(join(dir_in, fname)), compression='gzip')

        print("Converting text notes to lower case.")
        self.mimic_df['TEXT'] = self.mimic_df['TEXT'].str.lower()

    def label_sepsis(self, sepsis_df):

        # for every hadm_id in sepsis_df
        print("Getting all admission IDs in sepsis3 dataframe")
        hadm_id = sepsis_df['hadm_id']

        #add the corresponding entry as septic
        self.mimic_df['SEPTIC'] = np.where(self.mimic_df['HADM_ID'].isin(hadm_id), "True", "False")

    def keyword_filter(self):

        # keyword_set = {'pneumonia', 'empyema', 'meningitis', 'endocarditis', 'infection', 'altered mental status',
        #                'hyperthermia', 'hypothermia', 'tachardia', 'tachycardia', 'tachypnea',
        #                'leukocytosis', 'leukopenia', 'hyperglycemia',
        #                # 'urinary tract infection', 'abdominal infection', 'tissue infection', 'bone infection',
        #                # 'joint infection', 'wound infection', 'catheter infection',
        #                }

        keyword_pattern = "pneumonia | empyema | meningitis | endocarditis | infection | altered mental status | " \
                          "hyperthermia | hypothermia | tachardia | tachycardia | tachypnea | " \
                          "leukocytosis | leukopenia | hyperglycemia"


        septic_screening_notes = self.mimic_df[self.mimic_df['TEXT'].str.contains(keyword_pattern) == True]


        print("Number of unique hospital admissions after shortlisting: ", septic_screening_notes['HADM_ID'].nunique())

        print("Number of unique hospital admissions that are septic: ",
              septic_screening_notes.loc[septic_screening_notes['SEPTIC'] == "True"]['HADM_ID'].nunique())

        # print("Total number of shortlisted notes: ", len(septic_screening_notes))
        #
        # print("Sepsis distribution among the shortlisted notes (admissions are not unique): ",
        #       septic_screening_notes['SEPTIC'].value_counts())

        # print(septic_screening_notes.loc[septic_screening_notes['SEPTIC'] == "True"]["TEXT"])
        return septic_screening_notes


if __name__ == '__main__':

    dir_sepsis = '/home/madhumita/PycharmProjects/mimiciii-sepsis/sepsis3-data'
    dir_mimic = '/home/madhumita/dataset/mimiciii-v1.4'

    sepsis_data = SepsisData(dir_sepsis, dir_mimic)


