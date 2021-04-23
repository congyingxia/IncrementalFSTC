"""
This file is tosplit the dataset for the incremental fsl setting.
1, Split categories according to the split setting. All the categories (77)
   are splited into base (20), n1(10), n2(10), n3 (10), n4 (10), n5 (10)
   and ood(7).
2, Load training data and test data. Split training data into train and
   dev (20 for each class).
3, Split data for the incremental few shot setting. Sample few-shots (1 to 5)
   for new classes (n1 to n5).
"""

import os
import json
import random
import numpy as np


random.seed(42)
split_setting = {'base': 20, 'n1':10, 'n2':10, \
        'n3':10, 'n4':10, 'n5':10, 'ood':7}
TOTAL_CAT_NUM = 77


""" Load categories from json file and return a List.
"""
def load_categories(json_fname):
    f = open(json_fname, 'r')
    cat_list = json.load(f)
    return cat_list


""" Split categories into base, n1, ..., n5 and ood.
    Input: cat_list: a list of categories
           split_setting: a dict that saves category split setting
    Output: category split dict,
            {
            "base" => [list],
            "n1" => [list],
            ...,
            "n5" => [list],
            "ood" => [list],
            }
"""
def split_categories(cat_lst, split_setting):
    cat_split = {}
    for k, v in split_setting.items():
        # random sample v categories from cat_list
        cat_num = len(cat_lst)
        sample_idx = random.sample(list(np.arange(cat_num)), v)

        # add sampled categories into cat_split
        cat_split[k] = [cat_lst[i] for i in sample_idx]

        # remove sampled categories from cat_list
        new_cat_lst = []
        for i in range(cat_num):
            if i not in sample_idx:
                new_cat_lst.append(cat_lst[i])
        cat_lst = new_cat_lst

    if not os.path.isdir('./split'):
        os.makedirs('./split')
    dump_data(cat_split, 'split/category_split.txt')
    return cat_split


""" Check whether the categories are splited correctly.
"""
def check_split_categories(cat_split):
    merged = []
    for k, v in cat_split.items():
        #print("type:", k, "categories:", v)
        merged += v
    assert len(merged) == TOTAL_CAT_NUM


""" Load labeled data from file
    Input: fname
    Output: data, a dict of labeled samples
            key: categories
            value: a list of utterances
"""
def load_data(fname):
    f = open(fname, 'r')
    lines = f.readlines()[1:]
    data = {}
    for line in lines:
        arr = line.strip().split(',')
        sen = ','.join(arr[:-1])
        cat = arr[-1]
        assert '_' in cat, "error category: %s" % cat

        if cat not in data:
            data[cat] = []
        data[cat].append(sen)
    #print(len(data.keys()))
    #for k, v in data.items():
    #    print(k, len(v))
    return data


""" Dump data into files.
"""
def dump_data(data, fname):
    f = open(fname, 'w')
    for k, v in data.items():
        for s in v:
            f.write(k+'\t'+s+'\n')


""" Split training data into train and development set.
    Random sample 20 examples for each class as the development set.
    Input: train_data (dict, cat => list of samples)
    Output: train_data and dev_data (dict, cat => list of samples)
"""
def split_dev(train_data):
    dev_data = {}
    for cat_type, sen_list in train_data.items():
        dev_sens = random.sample(sen_list, 20)
        dev_data[cat_type] = dev_sens

        train_data[cat_type] = []
        for sen in sen_list:
            if sen not in dev_sens:
                train_data[cat_type].append(sen)
    return train_data, dev_data


""" Split data for the incremental few-shot setting: base, n1, ..., n5, ood.
    Sample few-shots (1 to 5) in the training for new classes (n1 to n5).
"""
def split_incremental(data, cat_split, mode):
    for cat_type, cat_list in cat_split.items():
        type_data = {}
        if mode == 'train' and cat_type in ['n1', 'n2', 'n3', 'n4', 'n5']:
            # sample few-shots for new classes
            # There are 10 new classes in each round. 
            # Each k-shot (k = 1..5) will have two classes.
            for i in range(len(cat_list)):
                # sampled examples
                k = i % 5 + 1
                # category
                c = cat_list[i]
                type_data[c] = random.sample(data[c], k)
        else:
            # We save all the data in the dev and base-training
            for c in cat_list:
               type_data[c] = data[c]

        data_dir = 'split/' + cat_type
        if not os.path.isdir(data_dir):
            os.makedirs(data_dir)
        dump_data(type_data, data_dir + '/' + mode + '.txt')


""" data preprocess in the main function.
"""
def main():
    # load categories from json file
    cat_list = load_categories('categories.json')
    assert len(cat_list) == TOTAL_CAT_NUM

    # split categories
    cat_split = split_categories(cat_list, split_setting)
    check_split_categories(cat_split)

    # load training and test data
    train_data = load_data('train.csv')
    test_data = load_data('test.csv')
    dump_data(test_data, 'split/total_test.txt')

    # split training data into train and development set
    train_data, dev_data = split_dev(train_data)
    dump_data(train_data, 'split/total_train.txt')
    dump_data(dev_data, 'split/total_dev.txt')

    # split training/dev/test data into base, n1, ..., n5, ood
    split_incremental(train_data, cat_split, 'train')
    split_incremental(dev_data, cat_split, 'dev')
    split_incremental(test_data, cat_split, 'test')

if __name__ == '__main__':
    main()
