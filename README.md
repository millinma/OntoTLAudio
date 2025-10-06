# Class Ontology and Data Scale in Audio Transfer Learning 

This repository contains and reproduces the results for the paper "How Class Ontology and Data Scale Affect Audio Transfer Learning", currently under review for ICASSP 2026.

## Results

We provide the following pre-computed and reproducible results, as well as pre-trained model states:

- Pre-trained model states on subsets of the AudioSet ontology are to be downloaded [here](https://zenodo.org/records/17276871) to be stored in `model_states/`

- A summary of the pre-training and fine-tuning performances under `results/pretraining/summary/metric.csv` and `results/finetuning/summary/metric.csv`, respectively.

- Plots as shown in the paper under `plots/`

## Reproduction

The experiments are based on the [autrainer](https://github.com/autrainer/autrainer) toolkit. We used version 0.5.0 for this work, but newer versions should also be supported. To install autrainer and all dependent libraries in a virtual environment run 

```
cd path/to/repo
python -m venv venv
source venv venv/bin/activate
pip install autrainer==0.5.0
```
Afterwards, the reproduction consists of three steps, each of which can be skipped with the pre-computed results. 
- *Pre-training*: produces pre-trained model states and results of pre-training experiments. 
- *Fine-tuning*: produces results of the fine-tuning experiments as well fine-tuned model states (which are omitted here for memory reasons).
- *Analysis*: Produces the plots appearing in the paper.

### Pre-training
*Note:* As we cannot provide the [AudioSet](https://research.google.com/audioset/) files themselves, it is not recommended to reproduce the pre-training, as the continuously removal of youtube videos likely leads to a smaller amount of data compared to the data on which this work is based. Additionally, download, preprocessing and pre-training of model states takes a significant amount of time. Instead, we recommend to use the provided pre-training states. 

To reproduce the pre-training, run the following commands.

```
autrainer fetch -cn config_pretraining.yaml
autrainer preprocess -cn config_pretraining.yaml
autrainer train -cn config_pretraining.yaml
autrainer postprocess -cn results/finetuning
```

### Fine-tuning
To run the fine-tuning experiments, or fine-tune the pre-trained model states on other datasets (the latter needing updates to the config files), run

```
autrainer fetch -cn config_finetuning.yaml
autrainer preprocess -cn config_finetuning.yaml
autrainer train -cn config_finetuning.yaml
autrainer postprocess results/pretraining
```


### Analysis

To run the analysis execute the respective jupyter notebooks

- `plot_finetuning_results.ipynb`: creates the plots for the finetuning performance across different pre-training states w.r.t. samples and classes, respectively.

- `plot_pretraining_correlation.ipynb` creates the plots of pair-wise cosine distance across pre-training states.