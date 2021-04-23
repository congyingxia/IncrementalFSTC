import os
import json
import random
import numpy as np

random.seed(42)
R_CLASS_NUM = 10 # class num per round
BASE_TRAIN_NUM = 500
DEV_NUM = 20
TEST_NUM = 40

whole_data = json.load(open('whole_data.json'))

"""
Data split into different rounds.
BASE: wiki, 10 classes, train/dev/test examples: 5000/200/400
N1: wiki, 10 classes, train/dev/test examples: 30/200/400
N2: wiki, 10 classes, train/dev/test examples: 30/200/400
N3: wiki, 10 classes, train/dev/test examples: 30/200/400
N4: wiki, 10 classes, train/dev/test examples: 30/200/400
N5: semeval, 10 classes, train/dev/test examples: 30/200/400
OOD: pubmed, 10 classes, train/dev/test examples: 30/200/400
"""

def remove_idx(old_idx, used_idx):
    new_idx = []
    for i in old_idx:
        if i not in used_idx:
            new_idx.append(i)
    return new_idx



def split_data(domain, split, class_list, idx_list):
    """
    Split data according to the setting.
    For training, each base class has BASE_TRAIN_NUM examples,
    while exampels for new classes ranges from 1 to 5.
    For dev/test, each class has DEV_NUM/TEST_NUM exmaples.
    """
    train_data = {}
    dev_data = {}
    test_data = {}
    train_num = BASE_TRAIN_NUM

    for i, c in enumerate(idx_list):
        class_name = class_list[c]
        examples = whole_data[domain][class_name]
        random.shuffle(examples)

        if split != 'base':
            # for new rounds, training examples varies from 1 to 5
            train_num = i % 5 + 1

        train_data[class_name] = examples[ : train_num]
        dev_data[class_name] = examples[train_num : train_num + DEV_NUM]
        test_data[class_name] = examples[train_num + DEV_NUM : \
                train_num + DEV_NUM + TEST_NUM]
    split_dir = './split/' + split
    if not os.path.isdir(split_dir):
        os.makedirs(split_dir)
    json.dump(train_data, open(split_dir+'/train.txt', 'w'))
    json.dump(dev_data, open(split_dir+'/dev.txt', 'w'))
    json.dump(test_data, open(split_dir+'/test.txt', 'w'))


# for base, choose classes from wiki
wiki_classes = list(whole_data['wiki'].keys())
wiki_class_idx = list(np.arange(len(wiki_classes)))
base_class_idx = random.sample(wiki_class_idx, R_CLASS_NUM)
remain_class_idx = remove_idx(wiki_class_idx, base_class_idx)
split_data('wiki', 'base', wiki_classes, base_class_idx)

# for n1, n2, n3, n4, choose classes from wiki
for i in range(1, 5):
    class_idx = random.sample(remain_class_idx, R_CLASS_NUM)
    remain_class_idx = remove_idx(remain_class_idx, class_idx)
    split_data('wiki', 'n' + str(i), wiki_classes, class_idx)

# for n5, choose classes from semeval
# for ood, choose classes from pubmed
domains = ['semeval', 'pubmed']
splits = ['n5', 'ood']
for i in range(2):
    domain = domains[i]
    split = splits[i]
    classes = list(whole_data[domain].keys())
    class_idx = list(np.arange(len(classes)))
    class_idx = random.sample(class_idx, R_CLASS_NUM)
    split_data(domain, split, classes, class_idx)
