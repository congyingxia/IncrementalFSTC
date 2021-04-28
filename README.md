# Incremental Few-shot Text Classification with Multi-round New Classes: Formulation, Dataset and System

This repository provides code and data for the following paper: [Incremental Few-shot Text Classification with Multi-round New Classes: Formulation, Dataset and System](https://arxiv.org/abs/2104.11882). Congying Xia*, Wenpeng Yin*, Yihao Feng, Philip Yu. NAACL, 2021. (* indicates equal contribution.)

# Requirements
This repo is implemented with pytorch.
Python==3.6.12
Pytorch==0.4.1
Huggingface Transformers
tqdm
scicy

# Data Format
There are two datasets used in our paper: [Banking77](https://github.com/PolyAI-LDN/task-specific-datasets) (Intent Detection) and [FewRel](https://github.com/thunlp/FewRel) (Relation classification). We create two benchmarks based on these two datasets for the incremental few-shot text classification task. Each dataset is a folder under the ./data folder, where each sub-folder contains the split for different rounds, including base, n1, n2, n3, n4, n5 and ood classes. The script for generating these splits are provided in run_data_preprocess.sh.

./data
└── banking77
    ├── run_data_preprocess.sh
    └── split
        ├── base
        ├── n1    
        ├── n2
        ├── n3    
        ├── n4
        ├── n5
        └── ood

# Usage
There are three different settings in our experiments:
1) Relation Classification without base classes;
2) Intent detection without base classes; 
3) Intent detection with base classes.

Each setting has a sub-folder under the ./code foler. There are multiple models (our model and baselines) implemented for each setting.
To run each method, please find the script for that method under that setting.
For example, if you want to run our proposed method ENTAILMENT for intent detection without base classes, please go to ./code/No_base_Intent and run sh train.entailment.commands.sh

