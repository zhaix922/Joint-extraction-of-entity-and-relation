
from collections import defaultdict
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Tuple, Union


import numpy as np
import torch
from torch import nn
from torch.utils.data.dataset import Dataset
from transformers import PreTrainedTokenizer
from torchvision import transforms

from util import json_file_load



@dataclass
class InputBert:
    """
    A single set of features of data.
    Property names are the same names as the corresponding inputs to a BERT model.

    """
    input_ids: torch.tensor
    attention_mask: torch.tensor
    token_type_ids: torch.tensor
    labels: Optional[torch.tensor] = None

class BIORelDataset(Dataset):
    def __init__(self,
                 dataset_path: str,
                 tokenizer: PreTrainedTokenizer,
                 labels2id: Dict[str, int],
                 max_len_seq: int = 512,
                 device: str = "cpu",
                 bert_hugging: bool = True):
        super(BIORelDataset).__init__()
        self.bert_hugging = bert_hugging
        self.max_len_seq = max_len_seq
        self.labels2id = labels2id
        self.tokenizer = tokenizer
        self.dataset = json_file_load(dataset_path)
        self.pad_token_label_id = nn.CrossEntropyLoss().ignore_index
        self.device = device
        self.input_features = self.dataset2tensors()

    def __len__(self):
        return len(self.input_features)

    def __getitem__(self, i) -> Union[Dict[str, torch.tensor],
                                      Tuple[List[torch.tensor], torch.tensor]]:
        if self.bert_hugging:
            return asdict(self.input_features[i])
        else:
            inputs = asdict(self.input_features[i])
            labels = inputs.pop('labels')
            return list(inputs.values()), labels

    def dataset2tensors(self):
        features = []
        for data in self.dataset:
            
            entity_list = []
            for relation in data["Relation"]:
                if relation["Relation_type"] == "CID":
                    head_entity_id = relation["Head_entity"]
                    head_entity = data["Entity"][head_entity_id]
                    tail_entity_id = relation["Tail_entity"]
                    tail_entity = data["Entity"][tail_entity_id]
                    entity_list.append((head_entity,"CID_S"))
                    entity_list.append((tail_entity,"CID_O"))
            entity_list.sort(key=lambda x:(x[0]["Sent_id"],x[0]["Pos"]))
            
            sentence_tokens = []
            sentence_labels = []
            entity_list_index = 0
            for i,sentence in enumerate(data["Sentences"]):
                flag = False
                sentence_split = []
                label_split = []
                while entity_list_index < len(entity_list) and i == entity_list[entity_list_index][0]["Sent_id"]:
                    # 带标签的句子还需要调试
                    flag = True
                    entity_type = entity_list[entity_list_index][1]
                    entity_start_pos,entity_tail_pos = entity_list[entity_list_index][0]["Pos"]
                    entity_start_pos += 1
                    entity_tail_pos += 1
                    pre_sentence = sentence[:entity_start_pos]
                    sentence_split.append(pre_sentence)
                    label_split.append("O")
                    entity_sentence = sentence[entity_start_pos:entity_tail_pos]
                    sentence_split.append(entity_sentence)
                    label_split.append(entity_type)
                    sentence = sentence[entity_tail_pos:]
                    entity_list_index += 1
                if not flag:
                    current_sentence_tokens = self.tokenizer(sentence)
                    current_sentence_labels = [self.labels2id["O"]]*len(current_sentence_tokens["input_ids"])
                    sentence_tokens.append(current_sentence_tokens)
                    sentence_labels.append(current_sentence_labels)
                else:
                    sentence_split.append(sentence)
                    label_split.append("O")
                    current_sentence_tokens = []
                    current_sentence_labels = []
                    for part_id in range(len(sentence_split)):
                        sentence_part = sentence_split[part_id]
                        sentence_label = label_split[part_id]
                        sentence_part_tokens = self.tokenizer.tokenize(sentence_part)
                        if sentence_label == 'O':
                            sentence_part_labels = [self.labels2id["O"]]*len(sentence_part_tokens)
                        else:
                            sentence_part_labels = [self.labels2id["B_" + sentence_label]] + [self.labels2id["I_" + sentence_label]]*(len(sentence_part_tokens) - 1)
                        current_sentence_tokens += sentence_part_tokens
                        current_sentence_labels += sentence_part_labels
                    current_sentence_tokens_id = self.tokenizer.convert_tokens_to_ids(current_sentence_tokens)
                    _current_sentence_tokens = {
                        "input_ids":[self.tokenizer.cls_token_id] + current_sentence_tokens_id + [self.tokenizer.sep_token_id],
                        "token_type_ids":[0]*(len(current_sentence_tokens_id) + 2),
                        "attention_mask":[1]*(len(current_sentence_tokens_id) + 2)
                    }
                    current_sentence_labels = [0] + current_sentence_labels + [0]
                    if len(current_sentence_labels) != len(_current_sentence_tokens["input_ids"]):
                        pass
                    sentence_tokens.append(_current_sentence_tokens)
                    sentence_labels.append(current_sentence_labels)
            for train_data,labels in zip(sentence_tokens,sentence_labels):
                features.append({
                    "train_data":dict(train_data),
                    "labels":labels
                })
        train_features = []
        for single_feature in features:
            train_data = single_feature["train_data"]
            input_ids = train_data["input_ids"]
            token_type_ids = train_data["token_type_ids"]
            attention_mask = train_data["attention_mask"]
            labels = single_feature["labels"]
            if len(input_ids) > self.max_len_seq:
                temp_tail = input_ids[-1]
                input_ids = input_ids[:self.max_len_seq - 1] + [temp_tail]
                temp_tail = token_type_ids[-1]
                token_type_ids = token_type_ids[:self.max_len_seq - 1] + [temp_tail]
                temp_tail = attention_mask[-1]
                attention_mask = attention_mask[:self.max_len_seq - 1] + [temp_tail]
                temp_tail = labels[-1]
                labels = labels[:self.max_len_seq - 1] + [temp_tail]
            elif len(input_ids) < self.max_len_seq:
                pad_len = self.max_len_seq - len(input_ids)
                input_ids += [self.tokenizer.pad_token_id]*pad_len
                token_type_ids += [0]*pad_len
                attention_mask += [0]*pad_len
                labels += [self.pad_token_label_id]*pad_len
            train_features.append((InputBert(
                input_ids = torch.tensor(input_ids,device = self.device),
                token_type_ids = torch.tensor(token_type_ids,device = self.device),
                attention_mask = torch.tensor(attention_mask,device = self.device),
                labels = torch.tensor(labels,device = self.device)
            )))
        return train_features



class SentencePairDataset(Dataset):
    def __init__(self,
                 dataset_path: str,
                 tokenizer: PreTrainedTokenizer,
                 labels2id: Dict[str, int],
                 max_len_seq: int = 512,
                 device: str = "cpu",
                 bert_hugging: bool = True):
        super(SentencePairDataset).__init__()
        self.bert_hugging = bert_hugging
        self.max_len_seq = max_len_seq
        self.labels2id = labels2id
        self.tokenizer = tokenizer
        self.dataset = json_file_load(dataset_path)
        self.device = device
        self.input_features = self.dataset2tensors()

    def __len__(self):
        return len(self.input_features)

    def __getitem__(self, i) -> Union[Dict[str, torch.tensor],
                                      Tuple[List[torch.tensor], torch.tensor]]:
        inputs = self.input_features[i]
        labels = inputs['labels']
        return inputs, labels
    @staticmethod
    def find_sentnce_idx(value_set,value):
        for i,sv in enumerate(value_set):
            if sv == value:
                return i
        return -1

    def dataset2tensors(self):
        features = []
        for data in self.dataset:
            
            sentence_pair = []
            for relation in data["Relation"]:
                source = relation["Source"]
                if '|' not in source:
                    continue
                sentence1,sentence2 = source.split('|')
                sentence_index1 = self.find_sentnce_idx(data["Sentences"],sentence1)
                sentence_index2 = self.find_sentnce_idx(data["Sentences"],sentence2)
                sentence_pair.append({
                    "Head_sentence_id":sentence_index1,
                    "Tail_sentence_id":sentence_index2,
                    "Relation_type":self.labels2id[relation["Relation_type"]]
                })
            sentence_tokens = []
            for i,sentence in enumerate(data["Sentences"]):
                current_sentence_tokens = self.tokenizer(sentence)
                for kw in current_sentence_tokens:
                    current_sentence_tokens[kw] = torch.tensor(current_sentence_tokens[kw])
                sentence_tokens.append(current_sentence_tokens)
            features.append({
                "inputs":sentence_tokens,
                "labels":sentence_pair
            })
        return features


class SentnecePairCollateFn(object):
    '''
    padding input (List[Example]) with same shape, then convert it to batch input.
    '''

    def __init__(self,num_labels:int = 2, training: bool = True):
        self.num_labels = 1 + (num_labels - 1)*2
        self.training = training

    def __call__(self, batch_list):
        # demo版：仅支持batch_size = 1
        data_feature = dict(
            training_input_ids = [],
            sentence_embeding_idx = [],
            training_token_type_ids = [],
            training_attention_mask = [],
            labels = [],
            sentence_pair_relation = []
        )
        for single_data in batch_list:
            training_input = single_data[0]
            sentence_pair_labels = single_data[1]
            sentence_inputs_ids = [single_sentence["input_ids"] for single_sentence in training_input["inputs"]]
            sentence_token_type_ids = [single_sentence["token_type_ids"] for single_sentence in training_input["inputs"]]
            sentence_attention_mask = [single_sentence["attention_mask"] for single_sentence in training_input["inputs"]]
            sentence_length = torch.tensor([len(sentence_input_id) for sentence_input_id in  sentence_inputs_ids])
            sentence_embeding_idx = sentence_length.cumsum(dim = 0)
            sentence_embeding_idx -= 1
            training_input_ids = torch.cat(sentence_inputs_ids,dim = 0)
            training_token_type_ids = torch.cat(sentence_token_type_ids,dim = 0)
            training_attention_mask = torch.cat(sentence_attention_mask,dim = 0)
            docuemnt_length = len(sentence_inputs_ids)
            labels = torch.zeros(docuemnt_length,docuemnt_length,dtype = torch.int)
            
            for sentence_pair in sentence_pair_labels:
                head_sentence_id = sentence_pair["Head_sentence_id"]
                tail_sentence_id = sentence_pair["Tail_sentence_id"]
                relation_type = sentence_pair["Relation_type"]
                labels[head_sentence_id][tail_sentence_id] = relation_type*2 - 1
                labels[tail_sentence_id][head_sentence_id] = relation_type*2
            
            sentence_pair_relation = torch.zeros(docuemnt_length,docuemnt_length,5)
            for i in range(docuemnt_length):
                for j in range(docuemnt_length):
                    if i == j - 1:
                        sentence_pair_relation[i][j][1] = 1
                    elif i == j + 1:
                        sentence_pair_relation[i][j][2] = 1
                    elif i < j - 1:
                        sentence_pair_relation[i][j][3] = 1
                    elif i > j + 1:
                        sentence_pair_relation[i][j][4] = 1

            data_feature["sentence_embeding_idx"].append(sentence_embeding_idx)
            data_feature["training_input_ids"].append(training_input_ids)
            data_feature["training_token_type_ids"].append(training_token_type_ids)
            data_feature["training_attention_mask"].append(training_attention_mask)
            data_feature["labels"].append(labels)
            data_feature["sentence_pair_relation"].append(sentence_pair_relation)
        
        for feature_key in data_feature:
            data_feature[feature_key] = torch.stack(data_feature[feature_key])
        return data_feature
    
