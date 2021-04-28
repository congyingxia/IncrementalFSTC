# Incremental Few-shot Text Classification with Multi-round New Classes: Formulation, Dataset and System

This repository provides code and data for the following paper: [Incremental Few-shot Text Classification with Multi-round New Classes: Formulation, Dataset and System](https://arxiv.org/abs/2104.11882). Congying Xia*, Wenpeng Yin*, Yihao Feng, Philip Yu. [NAACL, 2021](https://2021.naacl.org/program/accepted/). (* indicates equal contribution.)

# Introduction
Text classification is usually studied by labeling natural language texts with relevant categories from a predefined set. In the real world, new classes might keep challenging the existing system with limited labeled data. The system should be intelligent enough to recognize upcoming new classes with a few examples. In this work, we define a new task in the NLP domain, <mark>incremental few-shot text classification</mark>, where the system incrementally handles multiple rounds of new classes. For each round, there is a batch of new classes with a few labeled examples per class. Two major challenges exist in this new task: (i) For the learning process, the system should incrementally learn new classes round by round without re-training on the examples of preceding classes; (ii) For the performance, the system should perform well on new classes without much loss on preceding classes. In addition to formulating the new task, we also release two benchmark datasets in the incremental few-shot setting: intent classification and relation classification. Moreover, we propose two entailment approaches, ENTAILMENT and HYBRID, which show promise for solving this novel problem.

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

