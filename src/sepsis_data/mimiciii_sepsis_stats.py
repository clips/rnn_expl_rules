import pandas as pd
from os.path import join

PATH_MIMICIII_SEPSIS = '/home/madhumita/sepsis_mimiciii_discharge/'


def get_sepsis_mention(note_subset, hadm_ids):
    n_mention_sepsis = 0
    for cur_hadm_id in hadm_ids:
        if 'sepsis' in note_subset[note_subset['HADM_ID'] == cur_hadm_id]['TEXT'].item():
            n_mention_sepsis += 1
            # print(note_subset[note_subset['HADM_ID'] == hadm_id]['TEXT'].item())
    return n_mention_sepsis


def get_sepsis_distribution(note_subset):
    septic_hadm_ids = list(note_subset[note_subset['SEPTIC'] == "septic"]['HADM_ID'])
    print("Number cases that mention sepsis: ",
          get_sepsis_mention(note_subset, septic_hadm_ids),
          " out of ", len(septic_hadm_ids), " septic cases")

    non_septic_hadm_ids = list(note_subset[note_subset['SEPTIC'] == "non_septic"]['HADM_ID'])
    print("Number of cases that mention sepsis: ",
          get_sepsis_mention(note_subset, non_septic_hadm_ids),
          " out of ", len(non_septic_hadm_ids), " non_septic cases")


def load_df():
    note_subset = pd.read_csv(join(PATH_MIMICIII_SEPSIS, "mimic_sepsis_subset_df.csv"))
    return note_subset


if __name__ == '__main__':
    note_subset = load_df()
    get_sepsis_distribution(note_subset)
