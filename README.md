## RNAFlow: RNA Structure & Sequence Design via Inverse Folding-Based Flow Matching ##

RNAFlow is a flow matching model for protein-conditioned RNA sequence-structure design. Its denoising network integrates an RNA inverse folding model and a pre-trained RosettaFold2NA network for generation of RNA sequences and structures.

#### Environment Setup ####

#### Running Inference ####

#### Processing Dataset ####

Pre-processed datasets for both splits are given at ```rnaflow/data/rf2na_dataset.pickle``` and ```rnaflow/data/seq_sim_dataset.pickle```

To preprocess the dataset on your own, from the main directory run:

```
python rnaflow/data/process_data.py --pdbbind_csv rnaflow/data/pdbbind_na.csv \
    --pdb_out_dir rnaflow/data/pdbs \
    --rf2na_test_csv rnaflow/data/rf2na_test_val_set.csv \
    --dataset rnaflow/data/rf2na_dataset.pickle
```

#### Running Train ####

To retrain RNAFlow, get RF2NA weights from https://files.ipd.uw.edu/dimaio/RF2NA_apr23.tgz