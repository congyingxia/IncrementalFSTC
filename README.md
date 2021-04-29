# Incremental Few-shot Text Classification with Multi-round New Classes: Formulation, Dataset and System

This repository provides code and data for the following paper: [Incremental Few-shot Text Classification with Multi-round New Classes: Formulation, Dataset and System](https://arxiv.org/abs/2104.11882). Congying Xia*, Wenpeng Yin*, Yihao Feng, Philip Yu. [NAACL, 2021](https://2021.naacl.org/program/accepted/). (* indicates equal contribution.)

# Introduction
<p align = "justify"> 
Text classification is usually studied by labeling natural language texts with relevant categories from a predefined set. In the real world, new classes might keep challenging the existing system with limited labeled data. The system should be intelligent enough to recognize upcoming new classes with a few examples. In this work, we define a new task in the NLP domain, <b><i>incremental few-shot text classification</i></b>, where the system incrementally handles multiple rounds of new classes. For each round, there is a batch of new classes with a few labeled examples per class. Two major challenges exist in this new task: (i) For the learning process, the system should incrementally learn new classes round by round without re-training on the examples of preceding classes; (ii) For the performance, the system should perform well on new classes without much loss on preceding classes. In addition to formulating the new task, we also release two benchmark datasets in the incremental few-shot setting: intent classification and relation classification. Moreover, we propose two entailment approaches, ENTAILMENT and HYBRID, which show promise for solving this novel problem.
</p>

# Requirements
This repository is implemented through the [Huggingface Transformers](https://github.com/huggingface/transformers) package. To use the program the following prerequisites need to be installed.
* Huggingface Transformers
* Python==3.6.12
* Pytorch==0.4.1
* tqdm
* scicy

# Data Format
In our paper, we release two benchmark datasets for the incremental few-shot text classification task: IFS-INTENT and IFS-RELATION. These two benchmarks are created based on two datasets: [Banking77](https://github.com/PolyAI-LDN/task-specific-datasets) (Intent Detection) and [FewRel](https://github.com/thunlp/FewRel) (Relation classification). Each dataset(```banking77``` and ```FewRel```) is a sub-folder under the ```./data``` folder, where we provide the splits for different rounds, including ```base```, ```n1```, ```n2```, ```n3```, ```n4```, ```n5``` and ```ood``` classes. 

The scripts for generating these splits are provided in ```run_data_preprocess.sh```. Generally, we randomly split the classes in the original dataset into a base group, 5 rounds of new classes and a group of out-of-distribution classes. Then we split the train/test examples provided by the original dataset into different rounds according to the splited classes. For each round, we provide the data for training (```train.txt```) and test (```test.txt```).

```
./data
└── banking77
    ├── run_data_preprocess.sh
    └── split
        ├── base
        │   ├── train.txt
        │   └── test.txt
        ├── n1    
        ├── n2
        ├── n3    
        ├── n4
        ├── n5
        └── ood
```

# Usage
There are three different settings in our experiments:
1) Relation Classification without base classes;
2) Intent detection without base classes; 
3) Intent detection with base classes.

Each setting has a sub-folder under the ./code foler. There are multiple models (our model and baselines) implemented for each setting.
To run each method, please find the script for that method under that setting.
For example, if you want to run our proposed method ENTAILMENT for intent detection without base classes, please go to ./code/No_base_Intent and run sh train.entailment.commands.sh


# Reference

If you find our code useful, please cite our paper.

```
@article{xia2021incremental,
  title={Incremental Few-shot Text Classification with Multi-round New Classes: Formulation, Dataset and System},
  author={Congying Xia and Wenpeng Yin and Yihao Feng and Philip Yu},
  journal={arXiv preprint arXiv:2104.11882},  
  year={2021}
}
```
