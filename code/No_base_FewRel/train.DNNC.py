# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""BERT finetuning runner."""

from __future__ import absolute_import, division, print_function
import json
import argparse
import csv
import logging
import os
import random
import sys
import codecs
import numpy as np
import torch
import torch.nn as nn
from collections import defaultdict
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from scipy.stats import beta
from torch.nn import CrossEntropyLoss, MSELoss
from scipy.special import softmax
# from scipy.stats import pearsonr, spearmanr
# from sklearn.metrics import matthews_corrcoef, f1_score

from transformers.tokenization_roberta import RobertaTokenizer
from transformers.optimization import AdamW
from transformers.modeling_roberta import RobertaModel#RobertaForSequenceClassification


logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

# from pytorch_transformers.modeling_bert import BertPreTrainedModel, BertModel
# import torch.nn as nn

bert_hidden_dim = 1024
pretrain_model_dir = 'roberta-large' #'roberta-large' , 'roberta-large-mnli', 'bert-large-uncased'

def store_transformers_models(model, tokenizer, output_dir, flag_str):
    '''
    store the model
    '''
    output_dir+='/'+flag_str
    # if not os.path.exists(output_dir):
    #     os.makedirs(output_dir)
    print('starting model storing....')
    # model.save_pretrained(output_dir)
    torch.save(model.state_dict(), output_dir)
    # tokenizer.save_pretrained(output_dir)
    print('store succeed')

class RobertaForSequenceClassification(nn.Module):
    def __init__(self, tagset_size):
        super(RobertaForSequenceClassification, self).__init__()
        self.tagset_size = tagset_size

        self.roberta_single= RobertaModel.from_pretrained(pretrain_model_dir)
        self.hidden_layer_0 = nn.Linear(bert_hidden_dim*3, bert_hidden_dim)
        self.hidden_layer_1 = nn.Linear(bert_hidden_dim*2, bert_hidden_dim)
        self.hidden_layer_2 = nn.Linear(bert_hidden_dim, bert_hidden_dim)
        self.single_hidden2tag = RobertaClassificationHead(bert_hidden_dim, tagset_size)

    def forward(self, input_ids, input_mask, span_a_mask, span_b_mask, span_a_mask_v2, span_b_mask_v2):
        # outputs_single = self.roberta_single(input_ids, input_mask, None)
        # hidden_states_single = outputs_single[1]#torch.tanh(self.hidden_layer_2(torch.tanh(self.hidden_layer_1(outputs_single[1])))) #(batch, hidden)



        outputs_single = self.roberta_single(input_ids, input_mask, None)
        output_last_layer_tensor3 = outputs_single[0] #(batch_size, sequence_length, hidden_size)`)
        span_a_reps = torch.sum(output_last_layer_tensor3*span_a_mask.unsqueeze(2), dim=1) #(batch, hidden)
        span_b_reps = torch.sum(output_last_layer_tensor3*span_b_mask.unsqueeze(2), dim=1) #(batch, hidden)
        combined_rep = torch.cat([span_a_reps, span_b_reps, span_a_reps*span_b_reps],dim=1) #(batch, 3*hidden)
        MLP_input_v1 = torch.tanh(self.hidden_layer_0(combined_rep))#(batch, hidden)


        span_a_reps_v2 = torch.sum(output_last_layer_tensor3*span_a_mask_v2.unsqueeze(2), dim=1) #(batch, hidden)
        span_b_reps_v2 = torch.sum(output_last_layer_tensor3*span_b_mask_v2.unsqueeze(2), dim=1) #(batch, hidden)
        combined_rep_v2 = torch.cat([span_a_reps_v2, span_b_reps_v2, span_a_reps_v2*span_b_reps_v2],dim=1) #(batch, 3*hidden)
        MLP_input_v2 = torch.tanh(self.hidden_layer_0(combined_rep_v2))#(batch, hidden)

        MLP_input = torch.cat([MLP_input_v1, MLP_input_v2], axis=1)

        hidden_states_single = torch.tanh(self.hidden_layer_2(torch.tanh(self.hidden_layer_1(MLP_input)))) #(batch, hidden)


        score_single = self.single_hidden2tag(hidden_states_single) #(batch, tag_set)
        return score_single



class RobertaClassificationHead(nn.Module):
    """wenpeng overwrite it so to accept matrix as input"""

    def __init__(self, bert_hidden_dim, num_labels):
        super(RobertaClassificationHead, self).__init__()
        self.dense = nn.Linear(bert_hidden_dim, bert_hidden_dim)
        self.dropout = nn.Dropout(0.1)
        self.out_proj = nn.Linear(bert_hidden_dim, num_labels)

    def forward(self, features):
        x = features#[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x



class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, span_a_left=None, span_a_right=None,span_a_left_v2=None, span_a_right_v2=None, text_b=None, span_b_left=None, span_b_right=None,span_b_left_v2=None, span_b_right_v2=None, label=None, premise_class=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.span_a_left = span_a_left
        self.span_a_right = span_a_right
        self.span_a_left_v2 = span_a_left_v2
        self.span_a_right_v2 = span_a_right_v2

        self.text_b = text_b
        self.span_b_left = span_b_left
        self.span_b_right = span_b_right
        self.span_b_left_v2 = span_b_left_v2
        self.span_b_right_v2 = span_b_right_v2
        self.label = label
        self.premise_class = premise_class


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids,span_a_mask, span_b_mask,span_a_mask_v2, span_b_mask_v2, label_id, premise_class_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.span_a_mask = span_a_mask
        self.span_b_mask = span_b_mask
        self.span_a_mask_v2 = span_a_mask_v2
        self.span_b_mask_v2 = span_b_mask_v2
        self.label_id = label_id
        self.premise_class_id = premise_class_id


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(unicode(cell, 'utf-8') for cell in line)
                lines.append(line)
            return lines

class RteProcessor(DataProcessor):
    """Processor for the RTE data set (GLUE version)."""

    def load_train(self, round_list):
        examples_list = []
        class_list_up_to_now = []
        round_indicator_up_to_now = []
        class_2_sentlist_upto_this_round = defaultdict(list)
        for round in round_list:
            '''first collect the class set in this round'''
            examples_this_round = []
            class_set_in_this_round = set()
            class_2_examplelist_this_round = defaultdict(list)
            filename = '../../data/FewRel/split/'+round+'/train.txt'
            # tup_list = []
            with open(filename) as json_file:
                dev_data = json.load(json_file)
                for class_name, example_list in dev_data.items():
                    for example in example_list:
                        sent = ' '.join(example.get('tokens'))
                        head_left = example.get('h')[2][0][0]
                        head_right = example.get('h')[2][0][-1]
                        tail_left = example.get('t')[2][0][0]
                        tail_right = example.get('t')[2][0][-1]
                        class_2_examplelist_this_round[class_name].append((sent, head_left, head_right, tail_left, tail_right))
                    class_set_in_this_round.add(class_name)
            json_file.close()
            class_2_sentlist_upto_this_round = {**class_2_sentlist_upto_this_round, **class_2_examplelist_this_round}
            class_list_up_to_now += list(class_set_in_this_round)
            round_indicator_up_to_now+=[round]*len(class_set_in_this_round)
            '''transform each example into entailment pair'''
            filename = '../../data/FewRel/split/'+round+'/train.txt'
            with open(filename) as json_file:
                dev_data = json.load(json_file)
                for class_name, example_list in dev_data.items():
                    for example in example_list:
                        sent = ' '.join(example.get('tokens'))
                        head_left = example.get('h')[2][0][0]
                        head_right = example.get('h')[2][0][-1]
                        tail_left = example.get('t')[2][0][0]
                        tail_right = example.get('t')[2][0][-1]
                        # tup_list.append((sent, class_name, head_left, head_right, tail_left, tail_right))
                        '''positive pair'''
                        for hypo_tuple in class_2_examplelist_this_round.get(class_name):
                            for _ in range(5):
                                examples_this_round.append( InputExample(guid=round, text_a=sent, span_a_left=head_left, span_a_right=head_right,span_a_left_v2=hypo_tuple[1], span_a_right_v2=hypo_tuple[2], text_b=hypo_tuple[0], span_b_left=tail_left, span_b_right=tail_right, span_b_left_v2=hypo_tuple[3], span_b_right_v2=hypo_tuple[4], label='entailment', premise_class=class_name))
                                examples_this_round.append( InputExample(guid=round, text_a=hypo_tuple[0], span_a_left=hypo_tuple[1], span_a_right=hypo_tuple[2],span_a_left_v2=head_left, span_a_right_v2=head_right, text_b=sent, span_b_left=hypo_tuple[3], span_b_right=hypo_tuple[4], span_b_left_v2=tail_left, span_b_right_v2=tail_right, label='entailment', premise_class=class_name))

                        '''negative pairs'''
                        negative_class_set = set(class_set_in_this_round)-set([class_name])
                        for negative_class in random.sample(list(negative_class_set), 1):
                            for hypo_tuple in class_2_examplelist_this_round.get(negative_class):
                                examples_this_round.append( InputExample(guid=round, text_a=sent, span_a_left=head_left, span_a_right=head_right,span_a_left_v2=hypo_tuple[1], span_a_right_v2=hypo_tuple[2], text_b=hypo_tuple[0], span_b_left=tail_left, span_b_right=tail_right, span_b_left_v2=hypo_tuple[3], span_b_right_v2=hypo_tuple[4], label='non-entailment', premise_class=class_name))

            json_file.close()
            examples_list.append(examples_this_round)
        return examples_list, class_list_up_to_now, round_indicator_up_to_now, class_2_sentlist_upto_this_round


    def load_dev_or_test(self, round_list, seen_classes, train_class_2_sentlist_upto_this_round, flag):
        examples_rounds = []
        for round in round_list:
            examples = []
            instance_size = 0
            filename = '../../data/FewRel/split/'+round+'/'+flag+'.txt'
            with open(filename) as json_file:
                dev_data = json.load(json_file)
                for class_name, example_list in dev_data.items():
                    for example in example_list:
                        sent = ' '.join(example.get('tokens'))
                        head_left = example.get('h')[2][0][0]
                        head_right = example.get('h')[2][0][-1]
                        tail_left = example.get('t')[2][0][0]
                        tail_right = example.get('t')[2][0][-1]
                        if round == 'ood':
                            class_name = 'ood'
                        for seen_class in seen_classes:
                            '''each example compares with all seen classes'''
                            for hypo_tuple in train_class_2_sentlist_upto_this_round.get(seen_class):
                                examples.append( InputExample(guid=flag, text_a=sent, span_a_left=head_left, span_a_right=head_right,span_a_left_v2=hypo_tuple[1], span_a_right_v2=hypo_tuple[2], text_b=hypo_tuple[0], span_b_left=tail_left, span_b_right=tail_right, span_b_left_v2=hypo_tuple[3], span_b_right_v2=hypo_tuple[4], label='entailment', premise_class=class_name))

            examples_rounds+=examples
        return examples_rounds



    def get_labels(self):
        'here we keep the three-way in MNLI training '
        return ["entailment", "not_entailment"]
        # return ["entailment", "neutral", "contradiction"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[1]
            text_b = line[2]
            label = line[-1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

def wordpairID_2_tokenpairID(sentence, wordindex_left, wordindex_right, full_token_id_list, tokenizer, sent_1=True):
    '''pls note that the input indices pair include the b in (a,b), but the output doesn't'''
    '''first find the position of [2,2]'''
    position_two_two = 0
    for i in range(len(full_token_id_list)):
        if full_token_id_list[i]==2 and full_token_id_list[i+1]==2:
            position_two_two = i
            break
    span = ' '.join(sentence.split()[wordindex_left: wordindex_right+1])
    if wordindex_left!=0:
        '''this span is the begining of the sent'''
        span=' '+span

    span_token_list = tokenizer.tokenize(span)
    span_id_list = tokenizer.convert_tokens_to_ids(span_token_list)
    # print('span:', span, 'span_id_list:', span_id_list)
    if sent_1:
        # for i in range(wordindex_left, len(full_token_id_list)-len(span_id_list)):
        for i in range(wordindex_left, position_two_two):
            if full_token_id_list[i:i+len(span_id_list)] == span_id_list:
                return i, i+len(span_id_list), span_token_list

        return None, None, span_token_list
    else:
        # print('position_two_two:', position_two_two)
        for i in range(position_two_two+2, len(full_token_id_list)):
            if full_token_id_list[i:i+len(span_id_list)] == span_id_list:
                return i, i+len(span_id_list), span_token_list

        return None, None, span_token_list

def convert_examples_to_features(examples, label_list, eval_class_list, max_seq_length,
                                 tokenizer, output_mode,
                                 cls_token_at_end=False,
                                 cls_token='[CLS]',
                                 cls_token_segment_id=1,
                                 sep_token='[SEP]',
                                 sep_token_extra=False,
                                 pad_on_left=False,
                                 pad_token=0,
                                 pad_token_segment_id=0,
                                 sequence_a_segment_id=0,
                                 sequence_b_segment_id=1,
                                 mask_padding_with_zero=True):
    """ Loads a data file into a list of `InputBatch`s
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """

    label_map = {label : i for i, label in enumerate(label_list)}
    class_map = {label : i for i, label in enumerate(eval_class_list)}

    features = []
    give_up = 0
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3". " -4" for RoBERTa.
            special_tokens_count = 4 if sep_token_extra else 3
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - special_tokens_count)
        else:
            # Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
            special_tokens_count = 3 if sep_token_extra else 2
            if len(tokens_a) > max_seq_length - special_tokens_count:
                tokens_a = tokens_a[:(max_seq_length - special_tokens_count)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids:   0   0   0   0  0     0   0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = tokens_a + [sep_token]
        if sep_token_extra:
            # roberta uses an extra separator b/w pairs of sentences
            tokens += [sep_token]
        segment_ids = [sequence_a_segment_id] * len(tokens)

        if tokens_b:
            tokens += tokens_b + [sep_token]
            segment_ids += [sequence_b_segment_id] * (len(tokens_b) + 1)

        if cls_token_at_end:
            tokens = tokens + [cls_token]
            segment_ids = segment_ids + [cls_token_segment_id]
        else:
            tokens = [cls_token] + tokens
            segment_ids = [cls_token_segment_id] + segment_ids

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
            segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
        else:
            input_ids = input_ids + ([pad_token] * padding_length)
            input_mask = input_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
            segment_ids = segment_ids + ([pad_token_segment_id] * padding_length)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        if output_mode == "classification":
            label_id = label_map[example.label]
        elif output_mode == "regression":
            label_id = float(example.label)
        else:
            raise KeyError(output_mode)

        span_a_left, span_a_right, span_a_token_list = wordpairID_2_tokenpairID(example.text_a, example.span_a_left, example.span_a_right, input_ids, tokenizer, sent_1=True)
        span_b_left, span_b_right, span_b_token_list = wordpairID_2_tokenpairID(example.text_a, example.span_b_left, example.span_b_right, input_ids, tokenizer, sent_1=True)

        span_a_left_v2, span_a_right_v2, span_a_token_list_v2 = wordpairID_2_tokenpairID(example.text_b, example.span_a_left_v2, example.span_a_right_v2, input_ids, tokenizer, sent_1=False)
        span_b_left_v2, span_b_right_v2, span_b_token_list_v2 = wordpairID_2_tokenpairID(example.text_b, example.span_b_left_v2, example.span_b_right_v2, input_ids, tokenizer, sent_1=False)

        # print('span_b_left, span_b_right, span_b_token_list:', span_b_left, span_b_right, span_b_token_list)
        if span_a_left is None or span_b_left is None or span_a_left_v2 is None or span_b_left_v2 is None:
            '''give up this pair'''
            give_up+=1
            span_a_left, span_a_right = 0,2
            span_b_left, span_b_right = 0,2
            span_a_left_v2, span_a_right_v2 = 0,2
            span_b_left_v2, span_b_right_v2 = 0,2
        #     continue
        # else:
        span_a_mask = [0]*len(input_ids)
        for i in range(span_a_left, span_a_right):
            span_a_mask[i]=1
        span_b_mask = [0]*len(input_ids)
        for i in range(span_b_left, span_b_right):
            span_b_mask[i]=1

        span_a_mask_v2 = [0]*len(input_ids)
        for i in range(span_a_left_v2, span_a_right_v2):
            span_a_mask_v2[i]=1
        span_b_mask_v2 = [0]*len(input_ids)
        for i in range(span_b_left_v2, span_b_right_v2):
            span_b_mask_v2[i]=1

        features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              span_a_mask = span_a_mask,
                              span_b_mask = span_b_mask,
                              span_a_mask_v2 = span_a_mask_v2,
                              span_b_mask_v2 = span_b_mask_v2,
                              label_id=label_id,
                              premise_class_id = class_map[example.premise_class]))
    print('give_up:', give_up)

    return features

def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()




def load_class_names():
    readfile = codecs.open('../../data/banking77/split/category_split.txt', 'r', 'utf-8')
    class_list = set()
    ood_classes = set()
    class_2_split = {}
    for line in readfile:
        parts = line.strip().split('\t')
        class_str = parts[1].strip()
        split_str = parts[0].strip()
        class_2_split[class_str] = split_str # base, n1, ...n5, ood
        class_list.add(class_str)
        if split_str == 'ood':
            ood_classes.add(class_str)
    print('class_list size:', len(class_list), ' it has ood class size:', len(ood_classes))
    return list(class_list), ood_classes, class_2_split


def examples_to_features(source_examples, label_list, eval_class_list, args, tokenizer, batch_size, output_mode, dataloader_mode='sequential'):
    source_features = convert_examples_to_features(
        source_examples, label_list, eval_class_list, args.max_seq_length, tokenizer, output_mode,
        cls_token_at_end=False,#bool(args.model_type in ['xlnet']),            # xlnet has a cls token at the end
        cls_token=tokenizer.cls_token,
        cls_token_segment_id=0,#2 if args.model_type in ['xlnet'] else 0,
        sep_token=tokenizer.sep_token,
        sep_token_extra=True,#bool(args.model_type in ['roberta']),           # roberta uses an extra separator b/w pairs of sentences, cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
        pad_on_left=False,#bool(args.model_type in ['xlnet']),                 # pad on the left for xlnet
        pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
        pad_token_segment_id=0)#4 if args.model_type in ['xlnet'] else 0,)

    dev_all_input_ids = torch.tensor([f.input_ids for f in source_features], dtype=torch.long)
    dev_all_input_mask = torch.tensor([f.input_mask for f in source_features], dtype=torch.long)
    dev_all_segment_ids = torch.tensor([f.segment_ids for f in source_features], dtype=torch.long)
    dev_all_span_a_mask = torch.tensor([f.span_a_mask for f in source_features], dtype=torch.float)
    dev_all_span_b_mask = torch.tensor([f.span_b_mask for f in source_features], dtype=torch.float)
    dev_all_span_a_mask_v2 = torch.tensor([f.span_a_mask_v2 for f in source_features], dtype=torch.float)
    dev_all_span_b_mask_v2 = torch.tensor([f.span_b_mask_v2 for f in source_features], dtype=torch.float)
    dev_all_label_ids = torch.tensor([f.label_id for f in source_features], dtype=torch.long)
    dev_all_premise_class_ids = torch.tensor([f.premise_class_id for f in source_features], dtype=torch.long)

    dev_data = TensorDataset(dev_all_input_ids, dev_all_input_mask, dev_all_segment_ids, dev_all_span_a_mask, dev_all_span_b_mask,dev_all_span_a_mask_v2, dev_all_span_b_mask_v2, dev_all_label_ids, dev_all_premise_class_ids)
    if dataloader_mode=='sequential':
        dev_sampler = SequentialSampler(dev_data)
    else:
        dev_sampler = RandomSampler(dev_data)
    dev_dataloader = DataLoader(dev_data, sampler=dev_sampler, batch_size=batch_size)


    return dev_dataloader

def store_transformers_models(model, tokenizer, output_dir, flag_str):
    '''
    store the model
    '''
    output_dir+='/'+flag_str
    # if not os.path.exists(output_dir):
    #     os.makedirs(output_dir)
    print('starting model storing....')
    # model.save_pretrained(output_dir)
    torch.save(model.state_dict(), output_dir)
    # tokenizer.save_pretrained(output_dir)
    print('store succeed')

def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--task_name",
                        default=None,
                        type=str,
                        required=True,
                        help="The name of the task to train.")
    ## Other parameters
    parser.add_argument("--cache_dir",
                        default="",
                        type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--round_name",
                        default="",
                        type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--train_batch_size",
                        default=16,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=64,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=1e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=50,
                        type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',
                        type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    parser.add_argument('--server_ip', type=str, default='', help="Can be used for distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="Can be used for distant debugging.")


    args = parser.parse_args()


    processors = {
        "rte": RteProcessor
    }

    output_modes = {
        "rte": "classification"
    }

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                            args.gradient_accumulation_steps))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")


    task_name = args.task_name.lower()

    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))

    round_name_2_rounds={'r1':['n1', 'ood'],
                         'r2':['n1', 'n2', 'ood'],
                         'r3':['n1', 'n2', 'n3', 'ood'],
                         'r4':[ 'n1', 'n2', 'n3','n4', 'ood'],
                         'r5':[ 'n1', 'n2', 'n3','n4', 'n5', 'ood']}




    model = RobertaForSequenceClassification(3)
    tokenizer = RobertaTokenizer.from_pretrained(pretrain_model_dir, do_lower_case=args.do_lower_case)
    model.load_state_dict(torch.load('../../data/MNLI_pretrained.pt'), strict=False)
    model.to(device)

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

    optimizer = AdamW(optimizer_grouped_parameters,
                             lr=args.learning_rate)

    processor = processors[task_name]()
    output_mode = output_modes[task_name]
    # banking77_class_list, ood_class_set, class_2_split = load_class_names()

    round_list = round_name_2_rounds.get(args.round_name)
    '''load training in list'''
    train_examples_list, train_class_list, train_class_2_split_list, class_2_sentlist_upto_this_round = processor.load_train(round_list[:-1]) # no odd training examples
    assert len(train_class_list) == len(train_class_2_split_list)
    # assert len(train_class_list) ==  20+(len(round_list)-2)*10
    '''dev and test'''
    dev_examples = processor.load_dev_or_test(round_list, train_class_list, class_2_sentlist_upto_this_round, 'dev')
    test_examples = processor.load_dev_or_test(round_list, train_class_list, class_2_sentlist_upto_this_round, 'test')
    print('train size:', [len(train_i) for train_i in train_examples_list], ' dev size:', len(dev_examples), ' test size:', len(test_examples))
    entail_class_list = ['entailment', 'non-entailment']
    eval_class_list = train_class_list+['ood']
    test_split_list = train_class_2_split_list+['ood']
    train_dataloader_list = []
    for train_examples in train_examples_list:
        train_dataloader = examples_to_features(train_examples, entail_class_list, eval_class_list, args, tokenizer, args.train_batch_size, "classification", dataloader_mode='random')
        train_dataloader_list.append(train_dataloader)
    dev_dataloader = examples_to_features(dev_examples, entail_class_list, eval_class_list, args, tokenizer, args.eval_batch_size, "classification", dataloader_mode='sequential')
    test_dataloader = examples_to_features(test_examples, entail_class_list, eval_class_list, args, tokenizer, args.eval_batch_size, "classification", dataloader_mode='sequential')

    '''training'''
    max_test_acc = 0.0
    max_dev_acc = 0.0
    for round_index, round in enumerate(round_list[:-1]):
        '''for the new examples in each round, train multiple epochs'''
        train_dataloader = train_dataloader_list[round_index]
        for epoch_i in range(args.num_train_epochs):
            for _, batch in enumerate(tqdm(train_dataloader, desc="train|"+round+'|epoch_'+str(epoch_i))):
                model.train()
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, _, span_a_mask, span_b_mask,span_a_mask_v2, span_b_mask_v2,label_ids, premise_class_ids = batch

                logits = model(input_ids, input_mask,span_a_mask, span_b_mask, span_a_mask_v2, span_b_mask_v2)
                loss_fct = CrossEntropyLoss()
                # print('label_ids:', label_ids)
                loss = loss_fct(logits.view(-1, 3), label_ids.view(-1))
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
        print('\t\t round ', round, ' is over...')

    '''evaluation'''
    model.eval()


    logger.info("***** Running test *****")
    logger.info("  Num examples = %d", len(test_examples))

    preds = []
    gold_class_ids = []
    for _, batch in enumerate(tqdm(test_dataloader, desc="test")):
        input_ids, input_mask, segment_ids, span_a_mask, span_b_mask,span_a_mask_v2, span_b_mask_v2, label_ids, premise_class_ids = batch
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        span_a_mask = span_a_mask.to(device)
        span_b_mask = span_b_mask.to(device)
        span_a_mask_v2 = span_a_mask_v2.to(device)
        span_b_mask_v2 = span_b_mask_v2.to(device)
        gold_class_ids+=list(premise_class_ids.detach().cpu().numpy())

        with torch.no_grad():
            logits = model(input_ids, input_mask, span_a_mask, span_b_mask, span_a_mask_v2, span_b_mask_v2)
        if len(preds) == 0:
            preds.append(logits.detach().cpu().numpy())
        else:
            preds[0] = np.append(preds[0], logits.detach().cpu().numpy(), axis=0)

    # print('preds:', preds)
    preds = softmax(preds[0],axis=1)
    pred_label_3way = np.argmax(preds, axis=1) #dev_examples, 0 means "entailment"
    pred_probs = list(preds[:,0]) #prob for "entailment" class: (#input, #seen_classe)
    assert len(pred_label_3way) == len(test_examples)
    assert len(pred_probs) == len(test_examples)
    assert len(gold_class_ids) == len(test_examples)


    pred_label_3way = np.array(pred_label_3way).reshape(len(test_examples)//len(train_class_list),len(train_class_list))
    pred_probs = np.array(pred_probs).reshape(len(test_examples)//len(train_class_list),len(train_class_list))
    gold_class_ids = np.array(gold_class_ids).reshape(len(test_examples)//len(train_class_list),len(train_class_list))
    '''verify gold_class_ids per row'''
    rows, cols = gold_class_ids.shape
    for row in range(rows):
        assert len(set(gold_class_ids[row,:]))==1
    gold_label_ids = list(gold_class_ids[:,0])
    pred_label_ids_raw = list(np.argmax(pred_probs, axis=1))
    pred_max_prob = list(np.amax(pred_probs, axis=1))
    pred_label_ids = []
    for idd, seen_class_id in enumerate(pred_label_ids_raw):
        if pred_label_3way[idd][seen_class_id]==0:
            pred_label_ids.append(seen_class_id)
        else:
            pred_label_ids.append(len(train_class_list))

    assert len(pred_label_ids) == len(gold_label_ids)
    acc_each_round = []
    for round_name_id in round_list:
        #base, n1, n2, ood
        round_size = 0
        rount_hit = 0
        if round_name_id != 'ood':
            for ii, gold_label_id in enumerate(gold_label_ids):
                if test_split_list[gold_label_id] == round_name_id:
                    round_size+=1
                    if gold_label_id == pred_label_ids[ii]:
                        rount_hit+=1
            acc_i = rount_hit/round_size
            acc_each_round.append(acc_i)
        else:
            '''ood acc'''
            gold_binary_list = []
            pred_binary_list = []
            for ii, gold_label_id in enumerate(gold_label_ids):
                gold_binary_list.append(1 if test_split_list[gold_label_id] == round_name_id else 0)
                pred_binary_list.append(1 if pred_label_ids[ii]==len(train_class_list) else 0)
            overlap = 0
            for i in range(len(gold_binary_list)):
                if gold_binary_list[i] == 1 and pred_binary_list[i]==1:
                    overlap +=1
            recall = overlap/(1e-6+sum(gold_binary_list))
            precision = overlap/(1e-6+sum(pred_binary_list))
            acc_i = 2*recall*precision/(1e-6+recall+precision)
            acc_each_round.append(acc_i)
    final_test_performance = acc_each_round
    print('final_test_performance:', final_test_performance)

if __name__ == "__main__":
    main()

'''

CUDA_VISIBLE_DEVICES=7 python -u train.entailment.baseline.py --task_name rte --do_train --do_lower_case --num_train_epochs 5 --train_batch_size 16 --eval_batch_size 64 --learning_rate 1e-6 --max_seq_length 64 --seed 42 --round_name 'r1'


'''
