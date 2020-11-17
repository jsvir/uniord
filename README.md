
Implementation of [Deep Ordinal Regression using Optimal Transport Loss and Unimodal Output Probabilities](https://arxiv.org/abs/2011.07607)

### Requirements

* PyTorch >= 1.6.0
* pytorch-lightning >= 1.0
* Python >= 3.7

### Getting started
To reproduce our experiments, update the config files in `egs` with your data dirs. Then run the next scripts:

1. Adience: `python scripts/run_adience_exps.py`
2. HCI: `python scripts/run_hci_exps.py`
3. DR: `python scripts/run_dr_exps.py`

### Adding new datasets:
1. Update the `dataset/dataset.py` with your new dataset.
2. Add a new ptl module in `modules/dataset_based.py`
3. Add new config in `egs`

###  Authors:
- Uri Shaham
- Jonathan Svirsky