This repository contains code for the paper: [Distilling neural networks into skipgram-level decision lists](https://arxiv.org/abs/2005.07111). By Madhumita Sushil, Simon Šuster and Walter Daelemans, 2020.

We do not provide the any of the sepsis evaluation dataset directly because it has been derived from the [MIMIC-III dataset](). To access these datasets, you would first need access to the MIMIC-III dataset. Please send us a proof of your MIMIC-III access on the email ID `madhumita.sushil.k@gmail.com`  and we would provide you with our version of the sepsis datasets. The corresponding scripts can be found under `src/datasets/scripts`

Please use `main.py` under `src` as the starting point to obtain explanation rules for the datasets. To obtain explanation rules for sepsis estimation or to generate the synthetic sepsis dataset, prior access to the MIMIC-III dataset is required. Parameters for different datasets are present under corresponding classes in this file.
For example, to obtain explanations for a supported dataset, run a command like:

`export PYTHONPATH=<parent_directory_path>/rnn_expl_rules`

Substitute `<parent_directory_path>` with the path to this repository.

`python3 src/main.py --dataset=sepsis_mimic --get_explanations`

Currently supported datasets include `sst2`, `sepsis-mimic`, `sepsis-mimic-discharge`. You can add your own datasets by following the example of sst2 under `main.py`.
The dataset class defines all the relevant parameters for training or loading an LSTM model and then obtaining its explanations.
This method can be extended to other classifiers by updating the model requirements under `classifiers` and `explanations` for both training models and obtaining gradients of that model outputs respectively.

To evaluate the explanation pipeline on the synthetic data, please refer to `synthetic_data_pipeline.py` under `src`.

