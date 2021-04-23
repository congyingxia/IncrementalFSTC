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
import math
from transformers.tokenization_roberta import RobertaTokenizer
from transformers.optimization import AdamW
from transformers.modeling_roberta import RobertaModel#RobertaForSequenceClassification

from torch.nn.parameter import Parameter

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
        # self.roberta_single.load_state_dict(torch.load('../../data/MNLI_pretrained.pt'), strict=False)
        # self.single_hidden2tag = RobertaClassificationHead(bert_hidden_dim, tagset_size)
        self.MLP = MLP(bert_hidden_dim)
        self.classWeight = Parameter(torch.Tensor(tagset_size, bert_hidden_dim))
        self.phi_avg = Parameter(torch.Tensor(1, bert_hidden_dim))
        self.phi_att = Parameter(torch.Tensor(1, bert_hidden_dim))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.classWeight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.phi_avg, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.phi_att, a=math.sqrt(5))

    def forward(self, input_ids, input_mask, output_rep=True):

        outputs_single = self.roberta_single(input_ids, input_mask, None)
        hidden_states_single = outputs_single[1]#torch.tanh(self.hidden_layer_2(torch.tanh(self.hidden_layer_1(outputs_single[1])))) #(batch, hidden)

        hidden_states_single_v2 = self.MLP(hidden_states_single) #(batch, tag_set)
        '''now, cosine with class weights'''
        normalized_input = hidden_states_single_v2/(1e-8+torch.sqrt(torch.sum(torch.square(hidden_states_single_v2), axis=1, keepdim=True)))
        normalized_weight = self.classWeight/(1e-8+torch.sqrt(torch.sum(torch.square(self.classWeight), axis=1, keepdim=True)))
        score_single =  normalized_input.matmul(normalized_weight.t()) #cosine
        if output_rep:
            return normalized_input #support, hidden
        else:
            return score_single


class ModelStageTwo(nn.Module):
    def __init__(self, tagset_size, roberta_model):
        super(ModelStageTwo, self).__init__()
        self.classWeight = Parameter(torch.Tensor(tagset_size, bert_hidden_dim))
        self.query_para = nn.Linear(bert_hidden_dim, bert_hidden_dim)
        self.key_para = nn.Linear(bert_hidden_dim, bert_hidden_dim)
        self.phi_avg = Parameter(torch.Tensor(1, bert_hidden_dim))
        self.phi_att = Parameter(torch.Tensor(1, bert_hidden_dim))
        self.γ = Parameter(torch.Tensor(1))
        self.reset_parameters(roberta_model)

    def reset_parameters(self, roberta_model):
        nn.init.kaiming_uniform_(self.phi_avg, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.phi_att, a=math.sqrt(5))
        self.classWeight = roberta_model.classWeight # it works

        #(input_ids, input_mask, model, novel_class_support_reps= novel_class_support_reps, base_class_mapping = original_base_class_idlist)
    def forward(self, input_ids, input_mask, roberta_model, novel_class_support_reps=None, fake_novel_size=None, base_class_mapping=None):
        '''rep for a query batch'''
        normalized_input = roberta_model(input_ids, input_mask, output_rep=True)#hidden_states_single_v2/(1e-8+torch.sqrt(torch.sum(torch.square(hidden_states_single_v2), axis=1, keepdim=True)))
        '''now, build class weights'''
        init_weight_for_all = self.classWeight[base_class_mapping] if base_class_mapping is not None else self.classWeight #the original order changed
        init_normalized_weight_for_all = init_weight_for_all/(1e-8+torch.sqrt(torch.sum(torch.square(init_weight_for_all), axis=1, keepdim=True)))

        '''the input are support examples for a fake novel class'''
        new_base_class_reps = init_normalized_weight_for_all[:-fake_novel_size] if fake_novel_size is not None else init_normalized_weight_for_all #(#base, hidden)
        new_novel_class_reps = []
        for supports_rep_per_class in novel_class_support_reps:
            '''supports_rep_per_class is normalized in roberta output already'''
            supports_rep_per_class_as_query = self.query_para(supports_rep_per_class)
            supports_rep_per_class_as_query = supports_rep_per_class_as_query/(1e-8+torch.sqrt(torch.sum(torch.square(supports_rep_per_class_as_query), axis=1, keepdim=True)))
            new_base_class_reps_as_key = self.key_para(new_base_class_reps)
            new_base_class_reps_as_key = new_base_class_reps_as_key/(1e-8+torch.sqrt(torch.sum(torch.square(new_base_class_reps_as_key), axis=1, keepdim=True)))
            '''cosine(query, key)'''
            attention_matrix = nn.Softmax(dim=1)(self.γ*(supports_rep_per_class_as_query.matmul(new_base_class_reps_as_key.t()))) #support, #base
            attention_context = attention_matrix.matmul(new_base_class_reps) #supprt, hidden
            w_att = torch.mean(attention_context, axis=0, keepdim=True)
            w_avg = torch.mean(supports_rep_per_class, axis=0, keepdim=True)#prototype rep
            composed_rep_for_novel_class = self.phi_avg*w_avg + self.phi_att*w_att
            normalized_composed_rep_for_novel_class = composed_rep_for_novel_class/(1e-8+torch.sqrt(torch.sum(torch.square(composed_rep_for_novel_class), axis=1, keepdim=True)))
            new_novel_class_reps.append(normalized_composed_rep_for_novel_class)
        if fake_novel_size is not None:
            assert len(new_novel_class_reps) == fake_novel_size
        update_normalized_weight_for_all = torch.cat([new_base_class_reps]+new_novel_class_reps, axis=0) #(10, hidden)
        '''compute logits for the query batch'''
        score_single =  normalized_input.matmul(update_normalized_weight_for_all.t()) #cosine
        return score_single



class MLP(nn.Module):
    """wenpeng overwrite it so to accept matrix as input"""

    def __init__(self, bert_hidden_dim):
        super(MLP, self).__init__()
        self.dense = nn.Linear(bert_hidden_dim, bert_hidden_dim)
        self.dropout = nn.Dropout(0.1)
    def forward(self, features):
        x = features#[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        # x = self.dropout(x)
        # x = self.out_proj(x)
        return x



class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
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
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


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
        find_class_list = []
        examples = []
        for round in round_list:
            filename = '../../data/banking77/split/'+round+'/train.txt'
            readfile = codecs.open(filename, 'r', 'utf-8')
            for row in readfile:
                parts = row.strip().split('\t')
                assert len(parts)==2
                class_name = parts[0].strip()
                if class_name not in set(find_class_list):
                    find_class_list.append(class_name)
                example_str = parts[1].strip()
                examples.append(
                    InputExample(guid='train', text_a=example_str, text_b=None, label=class_name))
            readfile.close()
        return examples, find_class_list

    def load_base_support_examples(self, class_list, fake_novel_support_size):
        class_2_textlist = defaultdict(list)
        filename = '../../data/banking77/split/base/train.txt'
        readfile = codecs.open(filename, 'r', 'utf-8')
        for row in readfile:
            parts = row.strip().split('\t')
            assert len(parts)==2
            class_name = parts[0].strip()
            example_str = parts[1].strip()
            class_2_textlist[class_name].append(example_str)
        readfile.close()
        truncated_class_2_textlist = {}
        for classname, textlist in class_2_textlist.items():
            truncated_class_2_textlist[classname] = random.sample(textlist, fake_novel_support_size)

        '''build examples'''
        examples = []
        for classname in class_list:
            for example_str in truncated_class_2_textlist.get(classname):
                examples.append(
                    InputExample(guid='train', text_a=example_str, text_b=None, label=classname))
        assert len(examples) == fake_novel_support_size*len(class_list)
        return examples

    def load_support_all_rounds(self, round_list):
        class_2_examples = {}
        find_class_list = []
        base_class_list = set()
        for round in round_list:
            filename = '../../data/banking77/split/'+round+'/train.txt'
            readfile = codecs.open(filename, 'r', 'utf-8')
            for row in readfile:
                parts = row.strip().split('\t')
                assert len(parts)==2
                class_name = parts[0].strip()
                if class_name not in set(find_class_list):
                    find_class_list.append(class_name)
                if round == 'base':
                    base_class_list.add(class_name)
                example_str = parts[1].strip()
                input_example = InputExample(guid='train', text_a=example_str, text_b=None, label=class_name)
                examples = class_2_examples.get(class_name)
                if examples is None:
                    examples = []
                examples.append(input_example)
                class_2_examples[class_name] = examples

            readfile.close()
        '''select 5 examples for base classes'''
        assert len(base_class_list) ==  20
        for base_class in base_class_list:
            example_candidates = class_2_examples.get(base_class)
            random.shuffle(example_candidates)
            support_examples = example_candidates[:5]
            class_2_examples[base_class] = support_examples


        return class_2_examples, find_class_list

    def load_dev_or_test(self, round_list, flag):
        '''
        find_class_list: classes in training, i.e., seen classes
        '''
        # find_class_list = []
        examples = []
        for round in round_list:
            filename = '../../data/banking77/split/'+round+'/'+flag+'.txt'
            readfile = codecs.open(filename, 'r', 'utf-8')
            for row in readfile:
                parts = row.strip().split('\t')
                assert len(parts)==2
                class_name = parts[0].strip()
                # if class_name not in set(find_class_list):
                #     find_class_list.append(class_name)
                example_str = parts[1].strip()
                examples.append(
                    InputExample(guid='eval', text_a=example_str, text_b=None, label=class_name))
            readfile.close()
        return examples



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



def convert_examples_to_features(examples, label_list, max_seq_length,
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

    features = []
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


        features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              label_id=label_id))
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


def examples_to_features(source_examples, label_list, args, tokenizer, batch_size, output_mode, dataloader_mode='sequential'):
    source_features = convert_examples_to_features(
        source_examples, label_list, args.max_seq_length, tokenizer, output_mode,
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
    dev_all_label_ids = torch.tensor([f.label_id for f in source_features], dtype=torch.long)

    dev_data = TensorDataset(dev_all_input_ids, dev_all_input_mask, dev_all_segment_ids, dev_all_label_ids)
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
                        default=3.0,
                        type=float,
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

    round_name_2_rounds={'base':['base', 'ood'],
                         'r1':['base', 'n1', 'ood'],
                         'r2':['base', 'n1', 'n2', 'ood'],
                         'r3':['base', 'n1', 'n2', 'n3', 'ood'],
                         'r4':['base', 'n1', 'n2', 'n3','n4', 'ood'],
                         'r5':['base', 'n1', 'n2', 'n3','n4', 'n5', 'ood']}

    processor = processors[task_name]()
    output_mode = output_modes[task_name]
    banking77_class_list, ood_class_set, class_2_split = load_class_names()

    round_list = round_name_2_rounds.get(args.round_name)
    train_examples, base_class_list = processor.load_train(['base']) #train on base only
    '''train the first stage'''
    model = RobertaForSequenceClassification(len(base_class_list))
    tokenizer = RobertaTokenizer.from_pretrained(pretrain_model_dir, do_lower_case=args.do_lower_case)
    model.to(device)

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

    optimizer = AdamW(optimizer_grouped_parameters,
                             lr=args.learning_rate)


    train_dataloader = examples_to_features(train_examples, base_class_list, args, tokenizer, args.train_batch_size, "classification", dataloader_mode='random')
    mean_loss = 0.0
    count =0
    for _ in trange(int(args.num_train_epochs), desc="Stage1Epoch"):
        for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
            model.train()
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch


            logits = model(input_ids, input_mask, output_rep=False)
            loss_fct = CrossEntropyLoss()

            loss = loss_fct(logits.view(-1, len(base_class_list)), label_ids.view(-1))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            mean_loss+=loss.item()
            count+=1
            # if count % 50 == 0:
            #     print('mean loss:', mean_loss/count)
    print('stage 1, train supervised classification on base is over.')
    '''now, train the second stage'''
    model_stage_2 = ModelStageTwo(len(base_class_list), model)
    model_stage_2.to(device)

    param_optimizer = list(model_stage_2.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    optimizer_stage_2 = AdamW(optimizer_grouped_parameters,
                             lr=args.learning_rate)
    mean_loss = 0.0
    count =0
    best_threshold = 0.0
    for _ in trange(int(args.num_train_epochs), desc="Stage2Epoch"):
        '''first, select some base classes as fake novel classes'''
        fake_novel_size = 15
        fake_novel_support_size = 5
        '''for convenience, we keep shuffle the base classes, select the last 5 as fake novel'''
        original_base_class_idlist = list(range(len(base_class_list)))
        # random.shuffle(original_base_class_idlist)
        shuffled_base_class_list = [ base_class_list[idd]  for idd in original_base_class_idlist]
        fake_novel_classlist = shuffled_base_class_list[-fake_novel_size:]
        '''load their support examples'''
        base_support_examples = processor.load_base_support_examples(fake_novel_classlist, fake_novel_support_size)
        base_support_dataloader = examples_to_features(base_support_examples, fake_novel_classlist, args, tokenizer, fake_novel_support_size, "classification", dataloader_mode='sequential')

        novel_class_support_reps = []
        for _, batch in enumerate(base_support_dataloader):
            input_ids, input_mask, segment_ids, label_ids = batch
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            model.eval()
            with torch.no_grad():
                support_rep_for_novel_class = model(input_ids, input_mask, output_rep=True)
            novel_class_support_reps.append(support_rep_for_novel_class)
        assert len(novel_class_support_reps) == fake_novel_size
        print('Extracting support reps for fake novel is over.')
        '''retrain on query set to optimize the weight generator'''
        train_dataloader = examples_to_features(train_examples, shuffled_base_class_list, args, tokenizer, args.train_batch_size, "classification", dataloader_mode='random')
        best_threshold_list = []
        for _ in range(10): #repeat 10 times is important
            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                model_stage_2.train()
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids = batch


                logits = model_stage_2(input_ids, input_mask, model, novel_class_support_reps= novel_class_support_reps, fake_novel_size=fake_novel_size, base_class_mapping = original_base_class_idlist)
                # print('logits:', logits)
                loss_fct = CrossEntropyLoss()

                loss = loss_fct(logits.view(-1, len(base_class_list)), label_ids.view(-1))
                loss.backward()
                optimizer_stage_2.step()
                optimizer_stage_2.zero_grad()
                mean_loss+=loss.item()
                count+=1
                if count % 50 == 0:
                    print('mean loss:', mean_loss/count)
                scores_for_positive = logits[torch.arange(logits.shape[0]), label_ids.view(-1)].mean()
                best_threshold_list.append(scores_for_positive.item())

        best_threshold = sum(best_threshold_list) / len(best_threshold_list)

    print('stage 2 training over')

    '''
    start testing
    '''

    '''first, get reps for all base+novel classes'''
    '''support for all seen classes'''
    class_2_support_examples, seen_class_list = processor.load_support_all_rounds(round_list[:-1]) #no support set for ood
    assert seen_class_list[:len(base_class_list)] == base_class_list
    seen_class_list_size = len(seen_class_list)
    support_example_lists = [class_2_support_examples.get(seen_class)  for seen_class in seen_class_list if seen_class not in base_class_list]

    novel_class_support_reps = []
    for eval_support_examples_per_class in support_example_lists:
        support_dataloader = examples_to_features(eval_support_examples_per_class, seen_class_list, args, tokenizer, 5, "classification", dataloader_mode='random')
        single_class_support_reps = []
        for _, batch in enumerate(support_dataloader):
            input_ids, input_mask, segment_ids, label_ids = batch
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            model.eval()
            with torch.no_grad():
                support_rep_for_novel_class = model(input_ids, input_mask, output_rep=True)
            single_class_support_reps.append(support_rep_for_novel_class)
        single_class_support_reps = torch.cat(single_class_support_reps,axis=0)
        novel_class_support_reps.append(single_class_support_reps)
    print('len(novel_class_support_reps):', len(novel_class_support_reps))
    print('len(base_class_list):', len(base_class_list))
    print('len(seen_class_list):', len(seen_class_list))
    assert len(novel_class_support_reps)+len(base_class_list) ==  len(seen_class_list)
    print('Extracting support reps for all  novel is over.')
    test_examples = processor.load_dev_or_test(round_list, 'test')
    test_class_list = seen_class_list+list(ood_class_set)
    print('test_class_list:', len(test_class_list))
    print('best_threshold:', best_threshold )
    test_split_list = []
    for test_class_i in test_class_list:
        test_split_list.append(class_2_split.get(test_class_i))
    test_dataloader = examples_to_features(test_examples, test_class_list, args, tokenizer, args.eval_batch_size, "classification", dataloader_mode='sequential')
    '''test on test batch '''
    preds = []
    gold_label_ids = []
    for input_ids, input_mask, segment_ids, label_ids in test_dataloader:
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        label_ids = label_ids.to(device)
        gold_label_ids+=list(label_ids.detach().cpu().numpy())
        model_stage_2.eval()
        with torch.no_grad():
            logits = model_stage_2(input_ids, input_mask, model, novel_class_support_reps= novel_class_support_reps, fake_novel_size=None, base_class_mapping = None)
        # print('test logits:', logits)
        if len(preds) == 0:
            preds.append(logits.detach().cpu().numpy())
        else:
            preds[0] = np.append(preds[0], logits.detach().cpu().numpy(), axis=0)

    preds = preds[0]

    pred_probs = preds#softmax(preds,axis=1)
    pred_label_ids_raw = list(np.argmax(pred_probs, axis=1))
    pred_max_prob = list(np.amax(pred_probs, axis=1))

    pred_label_ids = []
    for i, pred_max_prob_i in enumerate(pred_max_prob):
        if pred_max_prob_i < best_threshold:
            pred_label_ids.append(seen_class_list_size) #seen_class_list_size means ood
        else:
            pred_label_ids.append(pred_label_ids_raw[i])

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
                    # print('gold_label_id:', gold_label_id, 'pred_label_ids[ii]:', pred_label_ids[ii])
                    if gold_label_id == pred_label_ids[ii]:
                        rount_hit+=1
            acc_i = rount_hit/round_size
            acc_each_round.append(acc_i)
        else:
            '''ood f1'''
            gold_binary_list = []
            pred_binary_list = []
            for ii, gold_label_id in enumerate(gold_label_ids):
                # print('gold_label_id:', gold_label_id, 'pred_label_ids[ii]:', pred_label_ids[ii])
                gold_binary_list.append(1 if test_split_list[gold_label_id] == round_name_id else 0)
                pred_binary_list.append(1 if pred_label_ids[ii]==seen_class_list_size else 0)
            overlap = 0
            for i in range(len(gold_binary_list)):
                if gold_binary_list[i] == 1 and pred_binary_list[i]==1:
                    overlap +=1
            recall = overlap/(1e-6+sum(gold_binary_list))
            precision = overlap/(1e-6+sum(pred_binary_list))

            acc_i = 2*recall*precision/(1e-6+recall+precision)
            acc_each_round.append(acc_i)

    print('\n\t\t test_acc:', acc_each_round)
    final_test_performance = acc_each_round

    print('final_test_performance:', final_test_performance)

if __name__ == "__main__":
    main()

'''

CUDA_VISIBLE_DEVICES=7 python -u train.dynamicFewShot.baseline.py --task_name rte --do_train --do_lower_case --num_train_epochs 1 --train_batch_size 32 --eval_batch_size 64 --learning_rate 1e-6 --max_seq_length 64 --seed 42 --round_name 'r1'


'''
