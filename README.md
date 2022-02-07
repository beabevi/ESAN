# Equivariant Subgraph Aggregation Networks (ESAN)

This repository contains the official code of the paper
**[Equivariant Subgraph Aggregation Networks](https://arxiv.org/abs/2110.02910) (ICLR 2022)**

<p align="center">
<img src=./symmetries.png width=50% height=50%>
</p>

## Install

First create a conda environment
```
conda env create -f environment.yml
```
and activate it
```
conda activate subgraph
```

## Prepare the data
Run
```bash
python data.py --dataset $DATASET
```
where `$DATASET` is one of the following:
* MUTAG
* PTC
* PROTEINS
* NCI1
* NCI109
* IMDB-BINARY
* IMDB-MULTI
* ogbg-molhiv
* ogbg-moltox21
* ZINC
* CSL
* EXP
* CEXP

## Run the models

To perform hyperparameter tuning, make use of `wandb`:

1. In `configs/` folder, choose the `yaml` file corresponding to the dataset and setting (deterministic vs sampling) of interest, say `<config-name>`. This file contains the hyperparameters grid.

2. Run
    ```bash
    wandb sweep configs/<config-name>
    ````
    to obtain a sweep id `<sweep-id>`

3. Run the hyperparameter tuning with
    ```bash
    wandb agent <sweep-id>
    ```
    You can run the above command multiple times on each machine you would like to contribute to the grid-search

4. Open your project in your wandb account on the browser to see the results:
    * For the TUDatasets, the CSL and the EXP/CEXP datasets, refer to `Metric/valid_mean` and `Metric/valid_std` to obtain the results.

    * For the ogbg datasets and the ZINC dataset, compute mean and std of `Metric/train_mean`, `Metric/valid_mean`, `Metric/test_mean` over the different seeds of the same configuration.
    Then, take the results corresponding to the configuration obtaining the best validation metric.


## Credits

For attribution in academic contexts, please cite

```
@inproceedings{bevilacqua2022equivariant,
title={Equivariant Subgraph Aggregation Networks},
author={Beatrice Bevilacqua and Fabrizio Frasca and Derek Lim and Balasubramaniam Srinivasan and Chen Cai and Gopinath Balamurugan and Michael M. Bronstein and Haggai Maron},
booktitle={International Conference on Learning Representations},
year={2022},
}
```
