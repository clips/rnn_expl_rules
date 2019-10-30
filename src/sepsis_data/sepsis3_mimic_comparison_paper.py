from os.path import join, realpath

import numpy as np
import pandas as pd

class SepsisData:

    def __init__(self, dir_sepsis, dir_mimic):

        sepsis_df = self.read_sepsis3_csv(dir_sepsis)
        print(list(sepsis_df.columns.values))

        self.read_mimic_notes(dir_mimic)
        self.label_sepsis(sepsis_df)

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
        print("Getting all admission IDs in sepsis3 dataframe which match the explicit criteria")
        hadm_id = sepsis_df[(sepsis_df['sepsis_explicit'] == 1)]['hadm_id']
        print(hadm_id)

        #add the corresponding entry as septic
        self.mimic_df['SEPTIC'] = np.where(self.mimic_df['HADM_ID'].isin(hadm_id), "True", "False")


if __name__ == '__main__':

    dir_sepsis = '/home/madhumita/PycharmProjects/mimiciii-rnn_expl_rules/sepsis3-data'
    dir_mimic = '/home/madhumita/dataset/mimiciii'

    sepsis_data = SepsisData(dir_sepsis, dir_mimic)


