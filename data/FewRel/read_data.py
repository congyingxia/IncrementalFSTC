import os
import json

type_dt = json.load(open('./data/pid2name.json'))


def clean_relation_name(r):
    r = r.lower()
    r = r.replace('_', ' ')
    r = r.replace('-', ' ')
    if '(e2,e1)' in r:
        r = r.replace('(e2,e1)', '')
        # reverse e2 e1
        r_list = r.split(' ')
        r = r_list[1] + ' ' + r_list[0]
    else:
        r = r.replace('(e1,e2)', '')
    return r


def load_json_data(fname):
    data = json.load(open(fname))
    ret_data = {}
    for r in data.keys():
        if r in type_dt:
            ret_data[type_dt[r][0]] = data[r]
        else:
            new_r = clean_relation_name(r)
            ret_data[new_r] = data[r]
    return ret_data


def read_whole_data():
    whole_data = {}
    for k in ['wiki', 'nyt', 'pubmed', 'semeval']:
        if k == 'wiki':
            data = load_json_data('./data/train_wiki.json')
        else:
            data = load_json_data('./data/val_' + k + '.json')
        whole_data[k] = data
    return whole_data


def main():
    whole_data = read_whole_data()
    json.dump(whole_data, open('whole_data.json', 'w'))


if __name__ == '__main__':
    main()
