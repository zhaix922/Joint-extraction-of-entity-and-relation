import os
os.environ['NCCL_P2P_LEVEL'] = 'NVL'
os.environ['OMP_NUM_THREADS'] = '24'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['NCCL_DEBUG'] = 'INFO'

import logging
import random
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Mapping
from easydict import EasyDict

from torch.optim import Adam
from torch.utils.data import DataLoader
from transformers import (get_linear_schedule_with_warmup,
                          BertForTokenClassification,
                          AutoTokenizer,
                          TrainingArguments,
                          HfArgumentParser,
                          set_seed)

from nn_utils.optimizers import get_optimizer_with_weight_decay
from nn_utils.trainer import BIORelTrainer
from util import json_file_load
from bm_gbd_dataset import BIORelDataset

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization.Don't set if you want to train a model from scratch."
            )
        },
    )
    tokenizer_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The tokenizer for weights initialization.Don't set if you want to train a model from scratch."
            )
        },
    ),
    model_save_path:Optional[str] = field(
        default='./BIORelModel'
    ),
    device:Optional[str] = field(
        default="auto",
        metadata={
            "choices": ["auto", "cuda", "cpu"],
        },
    )

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_path: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )

    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text/json file)."})
    eval_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text/json file)."})
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."}
    )
    label_config_path:Optional[str] = field(default=None)
    max_seq_length: Optional[int] = field(default=512)

@dataclass
class BioRelTrainingArguments(TrainingArguments):
    dropout: Optional[float] = field(default=0.1)
    modules_to_save: Optional[str] = field(default=None)

logger = logging.getLogger(__name__)

def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments,BioRelTrainingArguments))
    model_args,dataset_args,training_args= parser.parse_args_into_dataclasses()
    set_seed(training_args.seed)


    label2id = json_file_load(dataset_args.label_config_path)
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    # Create loaders for datasets

    dataset_args.train_file = os.path.join(dataset_args.dataset_path,dataset_args.train_file)
    dataset_args.eval_file = os.path.join(dataset_args.dataset_path,dataset_args.eval_file)
    dataset_args.validation_file = os.path.join(dataset_args.dataset_path,dataset_args.validation_file)
    
     
    training_dataset = BIORelDataset(dataset_path=dataset_args.train_file,
                              tokenizer=tokenizer,
                              labels2id=label2id,
                              max_len_seq=dataset_args.max_seq_length,
                              device = model_args.device)
    val_dataset = BIORelDataset(dataset_path=dataset_args.validation_file,
                     tokenizer=tokenizer,
                     labels2id=label2id,
                     max_len_seq=dataset_args.max_seq_length,
                     device = model_args.device)

    bio_rel_bert = BertForTokenClassification.from_pretrained(
        pretrained_model_name_or_path = model_args.model_name_or_path,
        hidden_dropout_prob=training_args.dropout,
        attention_probs_dropout_prob=training_args.dropout,
        num_labels=len(label2id),
        id2label={str(v): k for k, v in label2id.items()}
    )

    trainer = BIORelTrainer(
        model= bio_rel_bert,
        tokenizer = tokenizer,
        train_dataset=training_dataset,
        eval_dataset=val_dataset,
        args = training_args
        )
    trainer.train()

if __name__=="__main__":
    main()
